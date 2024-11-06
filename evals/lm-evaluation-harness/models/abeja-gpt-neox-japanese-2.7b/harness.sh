MODEL_ARGS="pretrained=abeja/gpt-neox-japanese-2.7b"
TASK="jcommonsenseqa-1.1-0.2,jnli-1.1-0.2,marc_ja-1.1-0.2,jsquad-1.1-0.2,xlsum_ja"
python main.py --model hf-causal --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "2,3,3,3,1" --device "cuda" --output_path "models/abeja-gpt-neox-japanese-2.7b/result.json"
