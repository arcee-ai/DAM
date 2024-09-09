import argparse
import torch
import os
import json
from modeling.dam import DAMLinearLayer, DAMEmbeddingLayer, DAMRMSNorm
from utils import find_linear_layers, find_embedding_layers, find_norm_layers
from transformers import AutoModelForCausalLM, AutoTokenizer
from glom import glom, Assign
from tqdm import tqdm
from huggingface_hub import HfApi

def fix_config(save_path, num_models, non_linearity, merge_embedding_layers, merge_layernorms, uses_base_model):

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
    data['dam_embedding_layer'] = merge_embedding_layers
    data['dam_layernorms'] = merge_layernorms
    data['uses_base_model'] = uses_base_model

    with open(config_path, 'w') as file:
        json.dump(data, file, indent=2)

    return config_path  # Return the updated config

def merge_models(base_model_id, model_ids, output_path, device, use_base_model, non_linearity, merge_embedding_layers, merge_layernorms, repo_id):

    print(f"Loading base model: {base_model_id}")
    merged_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.bfloat16, device_map=device)

    print("Loading models to merge:")
    models = [AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device) for model_id in tqdm(model_ids, desc="Loading models")]

    # If use_base_model is True, append the base model to the models list
    if use_base_model:
        models.append(merged_model)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)

    if merge_layernorms:
        norm_modules = find_norm_layers(merged_model)

        # Step 1: Identify all the norm layers in the merged model that need to be processed.
        for m in tqdm(norm_modules, desc="Merging layer norms"):

            # Step 2: For the current layer, gather the corresponding norm layers from each model that is being merged.
            modules = [glom(model, m) for model in models]

            # Step 3: Create a new DAMRMSNorm that will be used to combine the weights from the models.
            dam_layernorm = DAMRMSNorm(
                normalized_shape=modules[0].weight.shape[0],
                num_models=len(models),
                eps=modules[0].variance_epsilon,
                dtype=modules[0].weight.dtype,
                non_linearity=non_linearity  # Set non_linearity based on user input
            ).to(device)

            # Loop over each module (i.e., norm layer) from the models being merged
            for i, module in enumerate(modules):
                # Assign the weights from the current model's layer to the corresponding slot in DAMLayerNorm
                dam_layernorm.weights[i].data = module.weight.data
                
            # Create an assignment operation to replace the original norm layer with the merged DAMLayerNorm
            assign = Assign(m, dam_layernorm)
            
            # Apply the assignment to the merged model, effectively inserting the merged DAMLayerNorm in place of the original layer
            glom(merged_model, assign)

    if merge_embedding_layers:
        embedding_module = find_embedding_layers(merged_model)

        modules = [glom(model, embedding_module[0]) for model in models]

        dam_embedding_layer = DAMEmbeddingLayer(
            num_embeddings=modules[0].num_embeddings,
            embedding_dim=modules[0].embedding_dim,
            num_models=len(models),
            dtype=modules[0].weight.dtype,
            non_linearity=non_linearity  # Set non_linearity based on user input
        ).to(device)

        for i, module in enumerate(tqdm(modules, desc="Merging embedding layers")):
            dam_embedding_layer.embeddings[i].data = module.weight.data  # Corrected assignment

        assign = Assign(embedding_module[0], dam_embedding_layer)
        glom(merged_model, assign)

    # Step 1: Identify all the linear layers in the merged model that need to be processed.
    linear_modules = find_linear_layers(merged_model)

    # Step 2: Loop through each linear layer found in the base model.
    for m in tqdm(linear_modules, desc="Merging linear layers"):

        # Step 3: For the current layer, gather the corresponding linear layers from each model that is being merged.
        modules = [glom(model, m) for model in models]

        # Step 4: Create a new DAMLinearLayer that will be used to combine the weights from the models.
        #         - The input and output dimensions (in_features and out_features) are taken from the first model's layer.
        #         - The number of models being merged is passed to the DAMLinearLayer.
        #         - The presence of a bias term is checked and passed to the new layer.
        #         - The data type of the weights is set based on the first model's layer.

        # This line creates an instance of DAMLinearLayer, which is a specialized linear layer designed to merge 
        # the corresponding linear layers from multiple models into a single layer.
        dam_linearlayer = DAMLinearLayer(
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
            dam_linearlayer.weights[i].data = module.weight.data
            
            # If the layer has a bias, assign the bias from the current model's layer to the corresponding slot
            if module.bias is not None:
                dam_linearlayer.biases[i].data = module.bias.data

        # Create an assignment operation to replace the original linear layer with the merged DAMLinearLayer
        assign = Assign(m, dam_linearlayer)
        
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

    fixed_config_path = fix_config(output_path, num_models=len(models), non_linearity=non_linearity, merge_embedding_layers=merge_embedding_layers, merge_layernorms=merge_layernorms, uses_base_model=use_base_model)

    # push to the hub
    tokenizer.push_to_hub(repo_id)
    merged_model.push_to_hub(repo_id)

    # Upload the fixed config file to the hub
    api = HfApi()
    api.upload_file(
        path_or_fileobj=fixed_config_path,
        path_in_repo="config.json",
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"Merge complete. Merged model saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Merge multiple models into one DAM model.")
    parser.add_argument("base_model_id", help="ID of the base model (every weight except linear layers will be sourced from this model)")
    parser.add_argument("model_ids", nargs='+', help="IDs of the models to merge (for linear layers)")
    parser.add_argument("--output_path", help="Path to save the merged model")
    parser.add_argument("--device", default="cpu", help="Device to use for computation (e.g., 'cpu', 'cuda')")
    parser.add_argument("--merge_embedding_layers", action='store_true', help="Include embedding layers in the merging process")
    parser.add_argument("--merge_layernorms", action='store_true', help="Include layer normalization layers in the merging process")
    parser.add_argument("--use_base_model", action='store_true', help="Include base model's linear layers in the merging process")
    parser.add_argument("--non_linearity", choices=['tanh', 'sigmoid', 'relu', 'None'], default=None, help="Non-linearity to use in DAMLinearLayer")
    parser.add_argument("--repo_id", required=True, help="Repository ID to push the merged model to")

    args = parser.parse_args()

    merge_models(args.base_model_id, args.model_ids, args.output_path, args.device, args.use_base_model, args.non_linearity, args.merge_embedding_layers, args.merge_layernorms, args.repo_id)

if __name__ == "__main__":
    # os.environ['HF_TOKEN'] = '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'
    # os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
    # os.environ['HF_HOME'] = '/workspace/hf-cache'

    # Model and dataset details
    # cache_dir = "/workspace/hf-cache"
    main()


# python merge.py mistralai/Mistral-7B-v0.1 augmxnt/shisa-gamma-7b-v1  WizardLM/WizardMath-7B-V1.1 arcee-train/Abel-7B-002-truncated-embeds --device cpu --output_path ./merged_model --merge_embedding_layers --use_base_model --non_linearity tanh --merge_layernorms --repo_id arcee-train/pplist-merged-untrained-with-base-layernorm-embedding