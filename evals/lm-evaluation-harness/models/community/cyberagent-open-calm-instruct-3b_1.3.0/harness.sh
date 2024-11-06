#!/bin/bash
set -eu
PROJECT_DIR=""
MODEL_ARGS="pretrained=${PROJECT_DIR}/instruction_tuning/outputs/open-calm-instruct-3b_1.3.0,tokenizer=cyberagent/open-calm-3b"
TASK="jcommonsenseqa-1.1-0.3,jnli-1.1-0.3,marc_ja-1.1-0.3,jsquad-1.1-0.3,jaqket_v2-0.1-0.3,xlsum_ja-1.0-0.3,xwinograd_ja,mgsm-1.0-0.3"
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot "3,3,3,2,1,1,0,5" \
    --device "cuda" \
    --output_path "models/community/cyberagent-open-calm-instruct-3b_1.3.0/result.json"
