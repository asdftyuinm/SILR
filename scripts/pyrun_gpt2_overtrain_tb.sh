#!/bin/bash

# the same as scripts/run_gpt2_124M.sh but with PyTorch

# if you wish to train on just a single GPU, simply skip the torchrun part, i.e.
# python train_gpt2.py ... (all the other arguments the same)
runcommand="python3 train_gpt2_overtrain_tb.py"
for k in "d12" "d24" "d36"; do
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
BS=4
fi
opt=adamw
betas="0.9 0.95"
if [ $k = "d12" ]
then
lrlist=( 0.0003 0.0006 0.0012 0.0024 0.0048 0.0096 )
fi
if [ $k = "d24" ]
then
lrlist=( 0.00015 0.0003 0.0006 0.0012 0.0024 0.0048 )
fi
if [ $k = "d36" ]
then
lrlist=( 0.0003 0.0006 0.0012 0.0024 )
fi
for lr in ${lrlist[@]}; do
    ${runcommand}  \
    --input_bin "dev/data/fineweb10B/fineweb_train_*.bin" \
    --sample_every 0 \
    --write_tensors 0 \
    --model ${k} \
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
    --use_modulewise_lr \
    --alpha_positively_with_lr \
    --unbalancedlr_every 25 \
    --grad_alpha_metric grad \
    --num_grad_steps 0.05 \
    --grad_unbalancedlr_every 0.2 \
    --assign_func tb_linear_map \
    --lr_min_ratio 1 \
    --lr_max_ratio 3 \
    --opt_type ${opt} | tee outputovertrain${k}${opt}${j}${i}tempbalance${lr}.txt
done
done
