MODEL_ARGS="pretrained=huggyllama/llama-30b,use_accelerate=True,load_in_8bit=True"
TASK="jsquad-1.1-0.3,jcommonsenseqa-1.1-0.3,jnli-1.1-0.3,marc_ja-1.1-0.3"
python main.py --model hf-causal-experimental --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "2,3,3,3" --device "cuda" --output_path "models/llama/llama-30b/result.json" --batch_size 2  > models/llama/llama-30b/harness.out 2> models/llama/llama-30b/harness.err
