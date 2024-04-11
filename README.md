# Multitask Finetuning on Pretrained KG-enchanced LLM for Question Answering and Machine Reading Comprehension Tasks

This repo provides the source code & data of our paper "Multitask Finetuning on Pretrained KG-enchanced LLM for Question Answering and Machine Reading Comprehension Tasks".


<p align="center">
  <img src="./figs/model_arch.png" width="500" title="DRAGON finetune overview" alt="">
</p>



## Dependencies

Run the following commands to create a conda environment:
```bash
conda create -y -n dragon python=3.8
conda activate dragon
pip install torch==1.10.1+cu113 torchvision -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install transformers==4.10 wandb nltk spacy==2.1.6
python -m spacy download en
pip install scispacy==0.3.0
pip install torch-scatter==2.0.9 torch-sparse==0.6.12 torch-geometric==2.0.0 -f https://pytorch-geometric.com/whl/torch-1.10.1+cu113.html
```

## Acknowledgment
This repo is built upon the following works:
```
DRAGON: Deep Bidirectional Language-Knowledge Graph Pretraining
[https://github.com/snap-stanford/GreaseLM](https://github.com/michiyasunaga/dragon)
```
