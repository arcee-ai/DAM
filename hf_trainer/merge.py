import argparse
import torch
import os
import json
from modeling.dam import DAMLinearLayer
from utils import find_linear_layers, find_embedding_layers
from transformers import AutoModelForCausalLM, AutoTokenizer
from glom import glom, Assign
from tqdm import tqdm


def fix_config(save_path, num_models):
    config_path = os.path.join(save_path, 'config.json')
    with open(config_path, 'r') as file:
        data = json.load(file)

    data['model_type'] = "mergedmistral"
    data['architectures'][0] = 'MergedMistralForCausalLM'
    data['num_merged_models'] = num_models

    with open(config_path, 'w') as file:
        json.dump(data, file, indent=2)

def merge_models(base_model_id, model_ids, output_path, device):
    print(f"Loading base model: {base_model_id}")
    merged_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.bfloat16, device_map=device)
    
    print("Loading models to merge:")
    models = [AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device) for model_id in tqdm(model_ids, desc="Loading models")]

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)

    linear_modules = find_linear_layers(merged_model)

    for m in tqdm(linear_modules, desc="Merging layers"):
        modules = [glom(model, m) for model in models]

        dam_layer = DAMLinearLayer(
            in_features=modules[0].in_features,
            out_features=modules[0].out_features,
            num_models=len(models),
            bias=modules[0].bias is not None,
            dtype=modules[0].weight.dtype
        ).to(device)

        for i, module in enumerate(modules):
            glom(dam_layer, f'weight_{i}').data = module.weight.data
            if module.bias is not None:
                glom(dam_layer, f'bias_{i}').data = module.bias.data

        assign = Assign(m, dam_layer)
        glom(merged_model, assign)

    print(f"Saving merged model to {output_path}")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    fix_config(output_path, num_models=len(models))

    print(f"Merge complete. Merged model saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Merge multiple models into one DAM model.")
    parser.add_argument("base_model_id", help="ID of the base model (every weight except linear layers will be sourced from this model)")
    parser.add_argument("model_ids", nargs='+', help="IDs of the models to merge (for linear layers)")
    parser.add_argument("output_path", help="Path to save the merged model")
    parser.add_argument("--device", default="cpu", help="Device to use for computation (e.g., 'cpu', 'cuda')")

    args = parser.parse_args()

    merge_models(args.base_model_id, args.model_ids, args.output_path, args.device)

if __name__ == "__main__":
    main()