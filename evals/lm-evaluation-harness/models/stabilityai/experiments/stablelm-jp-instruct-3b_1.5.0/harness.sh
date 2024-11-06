#!/bin/bash

set -eu

PROJECT_DIR=""
MODEL_NAME="stablelm-jp-instruct-3b_1.5.0"
PRETRAINED="${PROJECT_DIR}/sft/checkpoints/${MODEL_NAME}"
TOKENIZER="${PROJECT_DIR}/tokenizers/nai-hf-tokenizer/,use_fast=False"
MODEL_ARGS="pretrained=${PRETRAINED},tokenizer=${TOKENIZER}"
TASK="jsquad-1.1-0.3,jcommonsenseqa-1.1-0.3,jnli-1.1-0.3,marc_ja-1.1-0.3"
OUTPUT_PATH="models/stablelm/${MODEL_NAME}/result.json"
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot "2,3,3,3" \
    --device "cuda" \
    --output_path $OUTPUT_PATH
