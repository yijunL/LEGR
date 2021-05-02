import torch
import torch.nn as nn
from opt_einsum import contract
from long_seq import process_long_input
from losses import ATLoss
import dgl
import torch.nn.functional as F

class DocREModel(nn.Module):
    def __init__(self, config, model, emb_size=768, block_size=64, num_labels=-1):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.loss_fnt = ATLoss()

#         self.activation = nn.Tanh()
        self.activation = nn.ReLU()
        self.dropout_rate=0.5
        self.head_extractor = nn.Linear(3 * self.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(3 * self.hidden_size, emb_size)
        self.edge_layer = RelEdgeLayer(node_feat=self.hidden_size, edge_feat=self.hidden_size,
                                       activation=self.activation, dropout=self.dropout_rate)
        
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)
        self.path_info_attention = Attention(self.hidden_size * 2, self.hidden_size * 4)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.path_info_mapping = nn.Linear(self.hidden_size * 4, self.hidden_size)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()   #[batch_size, num_heads, sequence_length, sequence_length]
        hss, tss, rss, glob_embss = [], [], [], []
        #hts  [batch_size,n_pair,2]
        
        for i in range(len(entity_pos)):   #[batch_size, n_entity,n_mention,2]
            entity_embs, entity_atts, glob_embs = [], [], []
            mx_mention=max(len(e) for e in entity_pos[i])
            for n_e ,e in enumerate(entity_pos[i]):
                e_emb, e_att = [], []
                
                if len(e) >= 1:
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])  
                            mean_att=attention[i, :, start + offset].mean(0)
                            mean_att=F.pad(mean_att,(0,1),'constant',0)
                            e_att.append(mean_att)
                        else:
                            e_emb.append((torch.zeros(self.config.hidden_size)).to(sequence_output))
                            e_att.append(torch.zeros(c+1).to(attention))
                    glob_embs.append(torch.logsumexp(torch.stack(e_emb, dim=0), dim=0))
                else:
                    glob_embs.append(torch.zeros(self.config.hidden_size).to(sequence_output))
                for dif in range(mx_mention-len(e)):
                    e_emb.append((torch.zeros(self.config.hidden_size)).to(sequence_output))
                    e_att.append(torch.zeros(c+1).to(attention))
                    entity_pos[i][n_e].append((c-offset,c-offset))
                e_emb=torch.stack(e_emb, dim=0)
                e_att=torch.stack(e_att, dim=0)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)
            glob_embs=torch.stack(glob_embs, dim=0)
            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]   new: [n_e, mx_mention, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]   new: [n_e, mx_mention, seq_len]

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])  #new: [n_pair, mx_mention, d]
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])   #new: [n_pair, mx_mention, seq_len]
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            
            new_pos=torch.tensor(entity_pos[i])[:,:,0]+offset
            new_pos=torch.where(new_pos>c, torch.full_like(new_pos, c), new_pos).to(sequence_output.device).long()
            #print(new_pos.shape) (n_e,mx_mention)
        
            new_pos_h=torch.index_select(new_pos, 0, ht_i[:, 0]) #[n_pair, mx_mention]
            new_pos_t=torch.index_select(new_pos, 0, ht_i[:, 1])

            new_pos_h=new_pos_h.repeat(1,mx_mention).reshape(-1,mx_mention) #[n_pair*mx_mention, mx_mention]
            new_pos_t=new_pos_t.repeat(1,mx_mention).reshape(-1,mx_mention)

            pair_hatt=h_att.reshape(-1, c+1).gather(1, new_pos_t).reshape(-1, mx_mention,mx_mention).mean(1)                            #[n_pair*mx_mention, seq_len]
            pair_tatt=t_att.reshape(-1, c+1).gather(1, new_pos_h).reshape(-1, mx_mention,mx_mention).mean(1)
            
            pair_hatt=pair_hatt/(pair_hatt.sum(1,keepdim=True)+1e-5)
            pair_tatt=pair_tatt/(pair_tatt.sum(1,keepdim=True)+1e-5)
            pair_hatt=pair_hatt.reshape(-1)
            pair_tatt=pair_tatt.reshape(-1) #[n_pair*mention]

            new_hs=(pair_tatt*(hs.reshape(-1,self.config.hidden_size).t())).t().reshape(-1,mx_mention,self.config.hidden_size)
            new_ts=(pair_hatt*(ts.reshape(-1,self.config.hidden_size).t())).t().reshape(-1,mx_mention,self.config.hidden_size)#weighted

            new_hs=new_hs.sum(1)
            new_ts=new_ts.sum(1)
            
            new_h_att=(pair_tatt*(h_att.reshape(-1,c+1).t())).t().reshape(-1,mx_mention,c+1).sum(1) #[n_pair, sql_len]
            new_t_att=(pair_hatt*(t_att.reshape(-1,c+1).t())).t().reshape(-1,mx_mention,c+1).sum(1)

            new_h_att=new_h_att*new_t_att
            new_h_att=new_h_att/(new_h_att.sum(1,keepdim=True)+1e-5)
            rs = contract("ld,rl->rd", sequence_output[i], new_h_att[:,0:c]) 
            glob_embss.append(glob_embs)
            hss.append(new_hs)
            tss.append(new_ts)
            rss.append(rs)
        glob_embss=torch.cat(glob_embss, dim=0)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return glob_embss, hss, tss, rss

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                entity_graphs=None,
                path_table=None,
                instance_mask=None,
                ):

        sequence_output, attention = self.encode(input_ids, attention_mask)   #[batch_size, sequence_length, hidden_size]，[batch_size, num_heads, sequence_length, sequence_length]
        entity_embs ,hs, ts, rs = self.get_hrt(sequence_output, attention, entity_pos, hts)

        entity_graph_big = dgl.batch(entity_graphs)
        
        self.edge_layer(entity_graph_big, entity_embs)
        entity_graphs = dgl.unbatch(entity_graph_big)
        path_infos=[]
        now=0
        for i in range(len(entity_graphs)):
            path_t = path_table[i]
            for j in range(len(hts[i])):
                h = hts[i][j][0]
                t = hts[i][j][1]
                if (h, t) in path_t:
                    v = [val  for val in path_t[(h , t )]]
                elif (t, h) in path_t:
                    v = [val  for val in path_t[(t , h )]]
                else:
                    print("path not found.")
                    assert 1==2

                middle_node_num = len(v)
                if(middle_node_num==0):
                    path_infos.append(torch.zeros((self.hidden_size*4)).to(sequence_output))
                    continue
                
                edge_ids = (entity_graphs[i].edge_ids([h for _ in range(middle_node_num)], v)).to(input_ids)
                forward_first = torch.index_select(entity_graphs[i].edata['h'], dim=0, index=edge_ids)
                edge_ids = (entity_graphs[i].edge_ids(v, [t for _ in range(middle_node_num)])).to(input_ids)
                forward_second = torch.index_select(entity_graphs[i].edata['h'], dim=0, index=edge_ids)

                edge_ids = (entity_graphs[i].edge_ids([t for _ in range(middle_node_num)], v)).to(input_ids)
                backward_first = torch.index_select(entity_graphs[i].edata['h'], dim=0, index=edge_ids)
                edge_ids = (entity_graphs[i].edge_ids(v, [h for _ in range(middle_node_num)])).to(input_ids)
                backward_second = torch.index_select(entity_graphs[i].edata['h'], dim=0, index=edge_ids)

                tmp_path_info = torch.cat((forward_first, forward_second, backward_first, backward_second), dim=-1)
                _, attn_value = self.path_info_attention(torch.cat((hs[now+j], ts[now+j]), dim=-1), tmp_path_info)
                path_infos.append(attn_value)
            now+=len(hts[i])
            entity_graphs[i].edata.pop('h')

        path_infos = torch.stack(path_infos,dim=0)
        path_infos =self.dropout(self.activation(self.path_info_mapping(path_infos)))
        
        hs = self.dropout(self.activation(self.head_extractor(torch.cat([hs, rs, path_infos], dim=1))))  
        ts = self.dropout(self.activation(self.tail_extractor(torch.cat([ts, rs, path_infos], dim=1))))

        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)   #[-1,12,64]
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)   #[-1,12,64]
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)  #[-1,12,64,1]*[-1,12,1,64]

        logits = self.bilinear(bl)

        output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels),)
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.float(), labels.float())
            output = (loss.to(sequence_output),) + output
        return output

class RelEdgeLayer(nn.Module):
    def __init__(self,
                 node_feat,
                 edge_feat,
                 activation,
                 dropout=0.0):
        super(RelEdgeLayer, self).__init__()
        self.node_feat = node_feat
        self.edge_feat = edge_feat
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.mapping = nn.Linear(node_feat * 2, edge_feat)

    def forward(self, g, inputs):
        # g = g.local_var()

        g.ndata['h'] = inputs  # [total_mention_num, node_feat]
        g.apply_edges(lambda edges: {
            'h': self.dropout(self.activation(self.mapping(torch.cat((edges.src['h'], edges.dst['h']), dim=-1))))})
        g.ndata.pop('h')


class Attention(nn.Module):
    def __init__(self, src_size, trg_size):
        super().__init__()
        self.W = nn.Bilinear(src_size, trg_size, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, trg, attention_mask=None):
        '''
        src: [src_size]
        trg: [middle_node, trg_size]
        '''

        score = self.W(src.unsqueeze(0).expand(trg.size(0), -1), trg)
        score = self.softmax(score)
        value = torch.mm(score.permute(1, 0), trg)

        return score.squeeze(0), value.squeeze(0)
