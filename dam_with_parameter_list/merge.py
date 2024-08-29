import argparse
import torch
import os
import json
from modeling.dam import DAMLinearLayer
from utils import find_linear_layers, find_embedding_layers
from transformers import AutoModelForCausalLM, AutoTokenizer
from glom import glom, Assign
from tqdm import tqdm
from huggingface_hub import HfApi

def fix_config(save_path, num_models, non_linearity):

    config_path = os.path.join(save_path, 'config.json')
    with open(config_path, 'r') as file:
        data = json.load(file)

    if data['model_type'] == "mistral":
        data['model_type'] = "mergedmistral"
        data['architectures'][0] = 'MergedMistralForCausalLM'
    elif data['model_type'] == "llama":
        data['model_type'] = "mergedllama"
        data['architectures'][0] = 'MergedLlamaForCausalLM'

    data['num_merged_models'] = num_models
    data['non_linearity'] = non_linearity

    with open(config_path, 'w') as file:
        json.dump(data, file, indent=2)

    return config_path  # Return the updated config

def merge_models(base_model_id, model_ids, output_path, device, use_base_model, non_linearity):

    print(f"Loading base model: {base_model_id}")
    merged_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.bfloat16, device_map=device)

    print("Loading models to merge:")
    models = [AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device) for model_id in tqdm(model_ids, desc="Loading models")]

    # If use_base_model is True, append the base model to the models list
    if use_base_model:
        models.append(merged_model)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)

    # Step 1: Identify all the linear layers in the merged model that need to be processed.
    linear_modules = find_linear_layers(merged_model)

    # Step 2: Loop through each linear layer found in the base model.
    for m in tqdm(linear_modules, desc="Merging layers"):

        # Step 3: For the current layer, gather the corresponding linear layers from each model that is being merged.
        modules = [glom(model, m) for model in models]

        # Step 4: Create a new DAMLinearLayer that will be used to combine the weights from the models.
        #         - The input and output dimensions (in_features and out_features) are taken from the first model's layer.
        #         - The number of models being merged is passed to the DAMLinearLayer.
        #         - The presence of a bias term is checked and passed to the new layer.
        #         - The data type of the weights is set based on the first model's layer.

        # This line creates an instance of DAMLinearLayer, which is a specialized linear layer designed to merge 
        # the corresponding linear layers from multiple models into a single layer.
        dam_layer = DAMLinearLayer(
            in_features=modules[0].in_features,
            out_features=modules[0].out_features,
            num_models=len(models),
            bias=modules[0].bias is not None,
            dtype=modules[0].weight.dtype,
            non_linearity=non_linearity  # Set non_linearity based on user input
        ).to(device)


        # Loop over each module (i.e., linear layer) from the models being merged
        for i, module in enumerate(modules):
            # Assign the weights from the current model's layer to the corresponding slot in DAMLinearLayer
            dam_layer.weights[i].data = module.weight.data
            
            # If the layer has a bias, assign the bias from the current model's layer to the corresponding slot
            if module.bias is not None:
                dam_layer.biases[i].data = module.bias.data

        # Create an assignment operation to replace the original linear layer with the merged DAMLinearLayer
        assign = Assign(m, dam_layer)
        
        # Apply the assignment to the merged model, effectively inserting the merged DAMLinearLayer in place of the original layer
        glom(merged_model, assign)

    # Function to count the number of parameters
    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        return total_params

    # Call the function and print the number of parameters
    num_params = count_parameters(merged_model)
    print(f"Total number of parameters: {num_params}")

    print(f"Saving merged model to {output_path}")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    fixed_config_path = fix_config(output_path, num_models=len(models), non_linearity=non_linearity)

    # push to the hub
    tokenizer.push_to_hub("arcee-ai/pplist-merged-untrained")
    merged_model.push_to_hub("arcee-train/pplist-merged-untrained-with-base")

    # Upload the fixed config file to the hub
    api = HfApi()
    api.upload_file(
        path_or_fileobj=fixed_config_path,
        path_in_repo="config.json",
        repo_id="arcee-train/pplist-merged-untrained-with-base",
        repo_type="model",
    )

    print(f"Merge complete. Merged model saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Merge multiple models into one DAM model.")
    parser.add_argument("base_model_id", help="ID of the base model (every weight except linear layers will be sourced from this model)")
    parser.add_argument("model_ids", nargs='+', help="IDs of the models to merge (for linear layers)")
    parser.add_argument("--output_path", help="Path to save the merged model")
    parser.add_argument("--device", default="cpu", help="Device to use for computation (e.g., 'cpu', 'cuda')")
    parser.add_argument("--use_base_model", action='store_true', help="Include base model's linear layers in the merging process")
    parser.add_argument("--non_linearity", choices=['tanh', 'sigmoid', 'relu', None], default=None, help="Non-linearity to use in DAMLinearLayer")

    args = parser.parse_args()

    merge_models(args.base_model_id, args.model_ids, args.output_path, args.device, args.use_base_model, args.non_linearity)

if __name__ == "__main__":
    # os.environ['HF_TOKEN'] = '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'
    # os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
    # os.environ['HF_HOME'] = '/workspace/hf-cache'

    # Model and dataset details
    # cache_dir = "/workspace/hf-cache"
    main()


# python merge.py mistralai/Mistral-7B-v0.1 augmxnt/shisa-gamma-7b-v1  WizardLM/WizardMath-7B-V1.1 arcee-train/Abel-7B-002-truncated-embeds --device cpu --output_path /workspace/merged_model --use_base_model --non_linearity tanh