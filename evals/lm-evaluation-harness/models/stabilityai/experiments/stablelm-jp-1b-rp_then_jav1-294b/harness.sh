#!/bin/bash
set -eu

if [ -z ${JP_LLM_PATH+x} ]; then
    echo "Error: The JP_LLM_PATH environment variable is not set"
    exit 1
fi

MODEL_ARGS="pretrained=$JP_LLM_PATH/hf_model/1b-rp_then_jav1-294b,tokenizer=$JP_LLM_PATH/tokenizers/nai-hf-tokenizer/,use_fast=False"
TASK="jcommonsenseqa-1.1-0.2,jnli-1.1-0.2,marc_ja-1.1-0.2,jsquad-1.1-0.2,jaqket_v2-0.1-0.2,xlsum_ja,xwinograd_ja,mgsm"
NUM_FEW_SHOTS="3,3,3,2,1,1,0,5"
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot $NUM_FEW_SHOTS \
    --device "cuda" \
    --output_path "models/stablelm/stablelm-jp-1b-rp_then_jav1-294b/result.json"
