# merge_models.py
import torch
from transformers import AutoModelForCausalLM
from dam import DAMLinearLayer, DAMEmbeddingLayer
from glom import glom, Assign
from utils import find_linear_layers, find_embedding_layers
import os 

# Environment variables
os.environ['HF_TOKEN'] = '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'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
os.environ['HF_HOME'] = '/workspace/.hf'

def merge_models(model_names, cache_dir=None, apply_to_embeddings=True):
    base_model = AutoModelForCausalLM.from_pretrained(model_names[0], torch_dtype=torch.bfloat16, device_map="auto")
    models = [AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="auto") for name in model_names[1:]]
    
    linear_modules = find_linear_layers(base_model)
    embedding_modules = find_embedding_layers(base_model) if apply_to_embeddings else []

    # Merging linear and embedding layers
    for name in linear_modules + embedding_modules:
        base_module = glom(base_model, name)
        model_modules = [glom(model, name) for model in models]

        if isinstance(base_module, torch.nn.Linear):
            merged_layer = DAMLinearLayer(
                base_linear=base_module,
                linear_modules=model_modules,
                device=base_model.device,
                dtype=base_module.weight.dtype
            )
        elif isinstance(base_module, torch.nn.Embedding):
            merged_layer = DAMEmbeddingLayer(
                base_embedding=base_module,
                embedding_modules=model_modules,
                device=base_model.device,
                dtype=base_module.weight.dtype
            )
        else:
            continue
        
        # Directly assign the merged layer back to the model
        assign = Assign(name, merged_layer)
        glom(base_model, assign)
    print(base_model)
    exit()
    # Save the final merged model
    base_model.save_pretrained("merged_model")
    print("Merged model saved successfully!")

if __name__ == "__main__":
    model_names = [
        "mistralai/Mistral-7B-v0.1",
        "augmxnt/shisa-gamma-7b-v1",
        "WizardLM/WizardMath-7B-V1.1",
        "GAIR/Abel-7B-002"
    ]
    merge_models(model_names)
