MODEL_ARGS="pretrained=meta-llama/Llama-2-13b-hf,use_accelerate=True"

TASK="jsquad-1.1-0.3,jcommonsenseqa-1.1-0.3,jnli-1.1-0.3,marc_ja-1.1-0.3,jaqket_v2-0.1-0.3,xlsum_ja-1.0-0.3,xwinograd_ja,mgsm-1.0-0.3"
python main.py --model hf-causal-experimental --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "2,3,3,3,1,1,0,5" --device "cuda" --output_path "models/llama2/llama2-13b/result.json" --batch_size 2  > models/llama2/llama2-13b/harness.out 2> models/llama2/llama2-13b/harness.err
