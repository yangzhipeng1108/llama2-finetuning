#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=2
fi
mkdir -p $OUTPUT

deepspeed main.py \
   --data_path Dahoas/rm-static \
   --model_name_or_path /root/daryl149/llama-2-7b-chat-hf/  \
   --tokenizer_name  merged_tokenizer_hf/ \
   --train_files utils/data_path/alpaca_gpt4_data_zh.json \
   --validation_files  utils/data_path/alpaca_gpt4_data_zh.json  \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --do_train True \
   --do_eval True \
   --max_eval_samples 800 \
   --max_seq_len 1024 \
   --block_size 1024 \
   --learning_rate 5e-5 \
   --weight_decay 0. \
   --num_train_epochs 1 \
   --gradient_accumulation_steps 8 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 100 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --offload \
   --deepspeed \
   --gradient_checkpointing \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
