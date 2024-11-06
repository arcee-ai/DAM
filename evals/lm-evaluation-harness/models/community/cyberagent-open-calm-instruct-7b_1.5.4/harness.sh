#!/bin/bash

PRETRAINED="${PROJECT_DIR}/sft/checkpoints/open-calm-instruct-7b_1.5.4/"
TOKENIZER="cyberagent/open-calm-7b"
MODEL_ARGS="pretrained=${PRETRAINED},tokenizer=${TOKENIZER}"
TASK="jsquad-1.1-0.3,jcommonsenseqa-1.1-0.3,jnli-1.1-0.3,marc_ja-1.1-0.3"
NUM_FEWSHOT="2,3,3,3"
OUTPUT_PATH="models/community/cyberagent-open-calm-instruct-7b_1.5.4/result.json"
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot $NUM_FEWSHOT \
    --device "cuda" \
    --output_path $OUTPUT_PATH
