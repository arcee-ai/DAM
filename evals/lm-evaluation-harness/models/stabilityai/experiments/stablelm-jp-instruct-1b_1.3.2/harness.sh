#!/bin/bash

PROJECT_DIR=""
PRETRAINED="${PROJECT_DIR}/sft/checkpoints/stablelm-jp-instruct-1b_1.3.2"
TOKENIZER="${PROJECT_DIR}/tokenizers/nai-hf-tokenizer/,use_fast=False"
MODEL_ARGS="pretrained=${PRETRAINED},tokenizer=${TOKENIZER}"
TASK="jsquad-1.1-0.3,jcommonsenseqa-1.1-0.3,jnli-1.1-0.3,marc_ja-1.1-0.3"
NUM_FEWSHOTS="2,3,3,3"
OUTPUT_PATH="models/stablelm/stablelm-jp-instruct-1b_1.3.2/result.json"
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot $NUM_FEWSHOTS \
    --device "cuda" \
    --output_path $OUTPUT_PATH
