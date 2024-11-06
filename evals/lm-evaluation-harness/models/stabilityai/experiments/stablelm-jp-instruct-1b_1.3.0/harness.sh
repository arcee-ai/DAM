#!/bin/bash
MODEL_DIR=""
TOKENIZER_DIR=""
MODEL_ARGS="pretrained=${MODEL_DIR}/stablelm-jp-instruct-1b_1.3.0/,tokenizer=${TOKENIZER_DIR}/nai-hf-tokenizer/,use_fast=False"
TASK="jsquad-1.1-0.3,jcommonsenseqa-1.1-0.3,jnli-1.1-0.3,marc_ja-1.1-0.3"
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot "2,3,3,3" \
    --device "cuda" \
    --output_path "models/stablelm-jp-instruct-1b_1.3.0/result.json"
