#!/bin/bash

# the same as scripts/run_gpt2_124M.sh but with PyTorch

# if you wish to train on just a single GPU, simply skip the torchrun part, i.e.
# python train_gpt2.py ... (all the other arguments the same)
# torchrun --standalone --nproc_per_node=8 train_gpt2.py \

runcommand="python3 printlwlr.py"
for k in "d12"; do
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
    ${runcommand} \
    --input_bin "dev/data/fineweb10B/fineweb_train_*.bin" \
    --model ${k} \
    --batch_size ${BS} \
    --sequence_length 1024 \
    --total_batch_size 524288 \
    --dtype bfloat16 \
    --compile 1 \
    --tensorcores 1 \
    --flash 1 \
    --num_iterations 100 \
    --zero_stage 0 \
    --overfit_single_batch 0
done
