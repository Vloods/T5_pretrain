# Multitask Finetuning on Pretrained KG-enchanced LLM for Question Answering and Machine Reading Comprehension Tasks

This repo provides the source code & data of our paper "Multitask Finetuning on Pretrained KG-enchanced LLM for Question Answering and Machine Reading Comprehension Tasks".


<p align="center">
  <img src="./figs/model_arch.png" width="500" title="Model finetune overview" alt="">
</p>



## 0. Dependencies
Installation guide for training Roberta/T5 with H/A 100  

Requirements: python3.8, cuda11.8, torch2.0.1, pyg2.4, transformers4.10

Run the following commands to create a conda environment:

```bash
mamba create -y -n multitask_finetune python=3.8
mamba activate multitask_finetune
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 pyg=2.4 pytorch-sparse=0.6.17 -c pytorch -c nvidia -c pyg
pip install transformers==4.10 wandb nltk spacy==2.1.6
python -m spacy download en
pip install scispacy==0.3.0
pip install sentencepiece
```

## 1. Download pretrained models

Download pretrained models and place files under `./models`

| Model  | Size | Pretraining Text | Pretraining Knowledge Graph | Download Link |
| ------------- | --------- | ---- | ---- | ---- |
| T5-base | 220M parameters | BookCorpus (filtered) | ConceptNet | link (will be updated) |
| RoBERTa | 360M parameters | BookCorpus (filtered) | ConceptNet | link (will be updated) |


## 2. Download data

Download all the preprocessed data from **here** (link will be updated).

## 2. How to train

If you would like to train model on single task, run: 
```
scripts/run_train__{qa/mrc/kgqa_dataset_name}.sh
```

For joint training, run (don't forget to specify the task flags):
```
scripts/run_train__joint.sh
```

**(Optional)** To pretrain model on your own data, you can run:
```
scripts/run_pretrain.sh
scripts/T5_run_pretrain.sh
```

## Acknowledgment
This repo is built upon the following works:
```
DRAGON: Deep Bidirectional Language-Knowledge Graph Pretraining
[https://github.com/snap-stanford/GreaseLM](https://github.com/michiyasunaga/dragon)
```
