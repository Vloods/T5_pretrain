#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=4
dt=`date '+%Y%m%d_%H%M%S'`


dataset="obqa"
dataset_mrc="csqa"
shift
encoder='roberta-large'
args=$@


elr="3e-5"
dlr="1e-3"
bs=128
mbs=2
unfreeze_epoch=2
k=5 #num of gnn layers
residual_ie=2
gnndim=200


encoder_layer=-1
max_node_num=200
seed=1
lr_schedule=warmup_linear
warmup_steps=100

n_epochs=40
max_epochs_before_stop=30
ie_dim=400


max_seq_len=100
ent_emb=data/cpnet/tzw.ent.npy
kg=cpnet
kg_vocab_path=data/cpnet/concept.txt
inhouse=False


info_exchange=true
ie_layer_num=1
resume_checkpoint=None
resume_id=None
sep_ie_layers=false
random_ent_emb=false

fp16=true
upcast=true

load_model_path=models/general_model.pt

echo "***** hyperparameters *****"
echo "dataset: $dataset"
echo "enc_name: $encoder"
echo "batch_size: $bs mini_batch_size: $mbs"
echo "learning_rate: elr $elr dlr $dlr"
echo "gnn: dim $gnndim layer $k"
echo "ie_dim: ${ie_dim}, info_exchange: ${info_exchange}"
echo "******************************"

save_dir_pref='runs'
mkdir -p $save_dir_pref
mkdir -p logs

run_name=dragon__${dataset}_${dataset_mrc}_ih_${inhouse}_load__elr${elr}_dlr${dlr}_W${warmup_steps}_b${bs}_ufz${unfreeze_epoch}_e${n_epochs}_sd${seed}__${dt}
log=logs/train__${run_name}.log.txt

###### Training ######
python3 -u dragon_joint.py \
    --dataset $dataset --dataset_mrc $dataset_mrc\
    --encoder $encoder -k $k --gnn_dim $gnndim -elr $elr -dlr $dlr -bs $bs --seed $seed -mbs ${mbs} --unfreeze_epoch ${unfreeze_epoch} --encoder_layer=${encoder_layer} -sl ${max_seq_len} --max_node_num ${max_node_num} \
    --n_epochs $n_epochs --max_epochs_before_stop ${max_epochs_before_stop} --fp16 $fp16 --upcast $upcast --use_wandb false \
    --save_dir ${save_dir_pref}/${dataset}/${run_name} --save_model 1 \
    --run_name ${run_name} \
    --load_model_path $load_model_path \
    --residual_ie $residual_ie \
    --ie_dim ${ie_dim} --info_exchange ${info_exchange} --ie_layer_num ${ie_layer_num} --resume_checkpoint ${resume_checkpoint} --resume_id ${resume_id} --sep_ie_layers ${sep_ie_layers} --random_ent_emb ${random_ent_emb} --ent_emb_paths ${ent_emb//,/ } --lr_schedule ${lr_schedule} --warmup_steps $warmup_steps -ih ${inhouse} --kg $kg --kg_vocab_path $kg_vocab_path \
    --data_dir data \
> ${log}
