#!/bin/bash
PROJECT_DIR=""
MODEL_ARGS="pretrained=${PROJECT_DIR}/hf_model/3b-ja50_rp50-700b,tokenizer=${PROJECT_DIR}/tokenizers/nai-hf-tokenizer/,use_fast=False"
TASK="jcommonsenseqa-1.1-0.1,jnli,marc_ja,jsquad-1.1-0.1,jaqket_v2-0.1-0.1,xlsum_ja,xwinograd_ja,mgsm"
NUM_FEW_SHOTS="3,3,3,2,1,1,0,5"
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot $NUM_FEW_SHOTS \
    --device "cuda" \
    --output_path "models/stablelm/stablelm-jp-3b-ja50_rp50-700b/result_template-0.1.json"
