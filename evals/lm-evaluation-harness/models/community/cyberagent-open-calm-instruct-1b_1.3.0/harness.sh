#!/bin/bash
set -eu
PROJECT_DIR=""
MODEL_ARGS="pretrained=${PROJECT_DIR}/instruction_tuning/outputs/open-calm-instruct-1b_1.3.0,tokenizer=cyberagent/open-calm-1b"
TASK="jsquad-1.1-0.3,jcommonsenseqa-1.1-0.3,jnli-1.1-0.3,marc_ja-1.1-0.3"
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot "2,3,3,3" \
    --device "cuda" \
    --output_path "models/open-calm-instruct-1b_1.3.0/result.json"
