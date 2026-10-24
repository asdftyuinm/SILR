#!/bin/bash

# the same as scripts/run_gpt2_124M.sh but with PyTorch

# if you wish to train on just a single GPU, simply skip the torchrun part, i.e.
# python train_gpt2.py ... (all the other arguments the same)
runcommand="python3 train_gpt2_overtrain.py"
for k in "d36"; do
if [ $k = "d12" ]
then
BS=16
fi
if [ $k = "d24" ]
then
BS=8
fi
if [ $k = "d36" ]
then
BS=8
fi
if [ $k = "d48" ]
then
BS=1
fi
opt=signumadam
betas="0.9 0.95"
for lrmode in none half; do 
if [ $k = "d12" ]
then
lrlist=( 0.00003 0.00006 0.00012 0.00024 0.00048 0.00096 )
fi
if [ $k = "d24" ]
then
lrlist=( 0.000015 0.00003 0.00006 0.00012 0.00024 0.00048 )
fi
if [ $k = "d36" ]
then
lrlist=( 0.000015 0.00003 0.00006 0.00012 0.00024 0.00048 )
fi
if [ $k = "d48" ]
then
lrlist=( 0.0000075 0.000015 0.00003 0.00006 )
fi
for lr in ${lrlist[@]}; do
    ${runcommand}  \
    --input_bin "dev/data/fineweb10B/fineweb_train_*.bin" \
    --sample_every 0 \
    --write_tensors 0 \
    --model ${k} \
    --lwlr_mode ${lrmode} \
    --batch_size ${BS} \
    --sequence_length 1024 \
    --total_batch_size 524288 \
    --overfit_batch_number 10 \
    --dtype bfloat16 \
    --compile 1 \
    --tensorcores 1 \
    --flash 1 \
    --num_iterations 1000 \
    --learning_rate_decay_frac 0.0 \
    --weight_decay 0.0 \
    --betas ${betas} \
    --zero_stage 0 \
    --learning_rate ${lr} \
    --opt_type ${opt} | tee outputovertrain${k}${opt}${j}${i}${lrmode}${lr}.txt
done
done
done
