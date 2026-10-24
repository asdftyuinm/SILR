#!/bin/bash

# the same as scripts/run_gpt2_124M.sh but with PyTorch

# if you wish to train on just a single GPU, simply skip the torchrun part, i.e.
# python train_gpt2.py ... (all the other arguments the same)

for k in "d12"; do
BS=16
runcommand="python3 train_gpt2.py"
traindir="dev/data/fineweb10B/fineweb_train_*.bin"
valdir="dev/data/fineweb10B/fineweb_val_*.bin"
iternum=18865
opt=ademamix
betas="0.9 0.999"
for lrmode in none half; do 
if [ $lrmode = "none" ]
then
    lr=0.0012
fi
if [ $lrmode = "half" ]
then
    lr=0.0024
fi
    ${runcommand}  \
    --input_bin "${traindir}" \
    --input_val_bin "${valdir}" \
    --val_loss_every 250 \
    --sample_every 0 \
    --output_dir pylog_gpt2_${k}${opt}${lrmode}${lr} \
    --write_tensors 0 \
    --model ${k} \
    --lwlr_mode ${lrmode} \
    --batch_size ${BS} \
    --sequence_length 1024 \
    --total_batch_size 524288 \
    --dtype bfloat16 \
    --compile 1 \
    --tensorcores 1 \
    --flash 1 \
    --num_iterations ${iternum} \
    --opt_type ${opt} \
    --weight_decay 0.0 \
    --betas ${betas} \
    --zero_stage 0 \
    --learning_rate ${lr} \
    --warmup_iters 700 \
    --learning_rate_decay_frac 0.0 \
    --grad_clip 1.0 \
    --overfit_single_batch 0
done
done
