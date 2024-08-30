import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import Dataset
from data_setup import setup_datasets_and_templates  # Assuming this function is in data_setup.py
from data_preprocessing import preprocess_data  # Assuming this function is in data_preprocessing.py

from torch.utils.data import DataLoader
from transformers import default_data_collator
import click
from tqdm import tqdm

# Environment variables
os.environ['HF_TOKEN'] = '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' #'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'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
os.environ['HF_HOME'] = '/workspace/hf-cache'

def load_models(model_ids, cache_dir):
    """
    Load additional models.

    Args:
        model_ids (list): List of model IDs to load.
        cache_dir (str): The directory to cache the models.

    Returns:
        dict: Dictionary of loaded models with keys as model identifiers.
    """
    models_dict = {}
    for idx, model_id in enumerate(model_ids, 1):
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, cache_dir=cache_dir)
        models_dict[f'model_{idx}'] = model
    return models_dict

def compute_and_save_topk_logits(models_dict, tokenized_dataset, device, batch_size=2, top_k=50):
    """
    Compute the top-K logits and their indices from corresponding models in models_dict for each batch in the tokenized dataset
    and save them as additional columns in the dataset. Release GPU memory after each model.

    Args:
        models_dict (dict): Dictionary of models to compute logits from.
        tokenized_dataset (Dataset): The tokenized dataset.
        device (torch.device): The device to perform computation on.
        batch_size (int): Batch size for processing.
        top_k (int): Number of top logits to keep.

    Returns:
        Dataset: The updated dataset with top-K logits and their indices added as columns.
    """
    for idx, (model_key, model) in tqdm(enumerate(models_dict.items(), start=1), desc="Model", leave=True):
        model = model.to(device)  # Ensure model is on the correct device
        model.eval()  # Set model to evaluation mode

        topk_logits_list = []
        topk_indices_list = []
        
        # Create a DataLoader for the dataset with the specified batch size
        dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=default_data_collator)

        for batch in tqdm(dataloader, desc="Batch", leave=False):
            input_ids = batch[f'input_ids_{idx}'].to(device)
            attention_mask = batch[f'attention_mask_{idx}'].to(device)
            
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                topk_logits, topk_indices = torch.topk(logits, top_k, dim=-1)  # Get top-K logits and their indices

            # Convert the top-K logits and indices to lists and add to the respective lists
            topk_logits_list.extend(topk_logits.cpu().numpy().tolist())
            topk_indices_list.extend(topk_indices.cpu().numpy().tolist())

        # Add the top-K logits and indices as new columns in the dataset
        tokenized_dataset = tokenized_dataset.add_column(f'topk_logits_model_{idx}', topk_logits_list)
        tokenized_dataset = tokenized_dataset.add_column(f'topk_indices_model_{idx}', topk_indices_list)

    return tokenized_dataset

@click.command()
@click.option("--k", type=int, default=128, help="Number of top logits to keep.")
def main(k):
    # Environment variables
    os.environ['HF_TOKEN'] = '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'
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
    os.environ['HF_HOME'] = '/workspace/hf-cache'

    # Model and dataset details
    base_model_name = "mistralai/Mistral-7B-v0.1"
    cache_dir = "/workspace/hf-cache"

    # Setup tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True, cache_dir=cache_dir)
    
    # List of dataset names to load
    dataset_names = [
        "p1atdev/ichikara-instruction:20231115-1",
        "microsoft/orca-math-word-problems-200k",
        "meta-math/MetaMathQA"
    ]

    # Load and template the datasets
    templated_datasets = setup_datasets_and_templates(tokenizer, dataset_names, example_count=1069)

    # Data preprocessing
    combined_dataset, tokenizer = preprocess_data(templated_datasets, base_model_name, cache_dir)

    # Load additional models
    model_ids = [
        "augmxnt/shisa-gamma-7b-v1",
        "WizardLM/WizardMath-7B-V1.1",
        "arcee-train/Abel-7B-002-truncated-embeds"
    ]
    models_dict = load_models(model_ids, cache_dir)

    # Compute logits for each model and input set and save as additional columns
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    combined_dataset_with_top_k_logits = compute_and_save_topk_logits(models_dict, combined_dataset, device, batch_size=4, top_k=k)

    print(combined_dataset_with_top_k_logits)

    # Save the dataset with logits to disk
    combined_dataset_with_top_k_logits.save_to_disk(f"./DAM_logits_k_{k}")

if __name__ == "__main__":
    main()