# LEGR
## Code for Localized Entity Representation and Global Path Reasoning (LEGR) model for document-level relation extraction.  

## Requirements
* Python (tested on 3.7.9)
* CUDA (tested on 10.2)
* [PyTorch](http://pytorch.org/) (tested on 1.6.0)
* [Transformers](https://github.com/huggingface/transformers) (tested on 4.0.1)
* numpy (tested on 1.19.2)
* [apex](https://github.com/NVIDIA/apex) (tested on 0.1)
* [opt-einsum](https://github.com/dgasmith/opt_einsum) (tested on 3.3.0)
* ujson
* tqdm
* dgl-cu102 (tested on 0.4.3)

## Dataset
We use DocRED dataset to train an evaluate our model.  The [DocRED](https://www.aclweb.org/anthology/P19-1074/) dataset can be downloaded following the instructions at [link](https://github.com/thunlp/DocRED/tree/master/data).

Dataset should be placed as follows:  
```
LEGR
 |-- dataset
 |    |-- docred
 |    |    |-- train_annotated.json        
 |    |    |-- train_distant.json
 |    |    |-- dev.json
 |    |    |-- test.json
 |-- meta
 |    |-- rel2id.json
```

## Training and Evaluation
* Run train.py to train the model.  
* You can also change the `--model_name_or_path` argument to change the pretrained model (BERT-base or RoBERTa-large).  
* You can save the model by setting the `--save_path` argument before training. The model correponds to the best dev results will be saved.  
* Set the `--load_path` argument as the save path of the model and run train.py again to evaluate the model and produce the result.json that you can submit to the Codalab for the evaluation in the test dataset.
