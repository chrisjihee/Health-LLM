#!/bin/bash

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
data_path="data/generated_data_json/PMData_readiness_train_all.json"
output_path="output/medalpaca-7b-tuned"

torchrun --nproc_per_node=8 --master_port=2023 medalpaca/train.py \
    --model "medalpaca/medalpaca-7b" \
    --data_path "$data_path" \
    --output_dir "$output_path" \
    --train_in_8bit False \
    --use_lora False \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --gradient_checkpointing True \
    --global_batch_size 128 \
    --per_device_batch_size 4 \
    --num_epochs 5
# --model "medalpaca/medalpaca-7b" --data_path "data/generated_data_json/PMData_readiness_train_all.json" --output_dir "output/medalpaca-7b-tuned" --train_in_8bit False --use_lora False --bf16 True --tf32 True --fp16 False --gradient_checkpointing True --global_batch_size 128 --per_device_batch_size 4 --num_epochs 5
