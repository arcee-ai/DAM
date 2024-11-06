#!/bin/bash
set -eu

PROJECT_DIR=""
PRETRAINED="${PROJECT_DIR}/sft/models/stablelm-jp-instruct-3b_1.3.0/"
TOKENIZER="${PROJECT_DIR}/tokenizers/nai-hf-tokenizer/,use_fast=False"
MODEL_ARGS="pretrained=${PRETRAINED},tokenizer=${TOKENIZER}"
TASK="jcommonsenseqa-1.1-0.3,jnli-1.1-0.3,marc_ja-1.1-0.3,jsquad-1.1-0.3,jaqket_v2-0.1-0.3,xlsum_ja-1.0-0.3,xwinograd_ja,mgsm-1.0-0.3"
NUM_FEWSHOT="3,3,3,2,1,1,0,4"

OUTPUT_PATH="models/stablelm/stablelm-jp-instruct-3b_1.3.0/result.json"
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot $NUM_FEWSHOT \
    --device "cuda" \
    --no_cache \
    --output_path $OUTPUT_PATH
