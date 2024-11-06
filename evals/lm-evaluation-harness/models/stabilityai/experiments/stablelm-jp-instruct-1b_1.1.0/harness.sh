#!/bin/bash
MODEL_DIR=""
TOKENIZER_DIR=""
MODEL_ARGS="pretrained=${MODEL_DIR}/stablelm-jp-instruct-1b_1.1.0/,tokenizer=${TOKENIZER_DIR}/nai-hf-tokenizer/,use_fast=False"
TASK="jcommonsenseqa-1.1-0.3,jnli-1.1-0.3,marc_ja-1.1-0.3,jsquad-1.1-0.3,jaqket_v2-0.1-0.3,xlsum_ja-1.0-0.3,xwinograd_ja,mgsm-1.0-0.3"
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot "3,3,3,2,1,1,0,5" \
    --device "cuda" \
    --output_path "models/stablelm/stablelm-jp-instruct-1b_1.1.0/result.json"
