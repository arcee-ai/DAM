#!/bin/bash
set -eu
PROJECT_DIR= # XXX set your own project dir
MODEL_ARGS="pretrained=${PROJECT_DIR}/hf_model/stablelm-ja-base-alpha-7b,tokenizer=${PROJECT_DIR}/tokenizers/nai-hf-tokenizer/,use_fast=False,trust_remote_code=True"
TASK="jcommonsenseqa-1.1-0.2,jnli-1.1-0.2,marc_ja-1.1-0.2,jsquad-1.1-0.2,jaqket_v2-0.1-0.2,xlsum_ja,xwinograd_ja,mgsm"
NUM_FEW_SHOTS="3,3,3,2,1,1,0,5"
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot $NUM_FEW_SHOTS \
    --device "cuda" \
    --output_path "models/stablelm/stablelm-ja-base-alpha-7b/result.json"
