import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import Dataset, load_from_disk
from data_setup import setup_datasets_and_templates  # Assuming this function is in data_setup.py
from data_preprocessing import preprocess_data  # Assuming this function is in data_preprocessing.py

from torch.utils.data import DataLoader
from transformers import default_data_collator
from pathlib import Path
import click
from tqdm import tqdm

os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

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

def compute_and_save_topk_logits(models_dict, tokenized_dataset, device, batch_size=2, top_k=50, compute_logits=True):
    """
    Compute the top-K logits and their indices from corresponding models in models_dict for each batch in the tokenized dataset
    and save them as additional columns in the dataset. Release GPU memory after each model.

    Args:
        models_dict (dict): Dictionary of models to compute logits from.
        tokenized_dataset (Dataset): The tokenized dataset.
        device (torch.device): The device to perform computation on.
        batch_size (int): Batch size for processing.
        top_k (int): Number of top logits to keep.
        compute_logits (bool): Whether to compute logits or just keep input_ids.

    Returns:
        Dataset: The updated dataset with top-K logits and their indices added as columns.
    """
    for model_key, model in tqdm(models_dict.items(), desc="Model", leave=True):
        idx = int(model_key.split('_')[-1])  # Extract the integer index from the model_key
        model = model.to(device)  # Ensure model is on the correct device
        model.eval()  # Set model to evaluation mode

        topk_logits_list = []
        topk_indices_list = []
        
        # Create a DataLoader for the dataset with the specified batch size
        dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=default_data_collator)

        for batch in tqdm(dataloader, desc="Batch", leave=False):
            input_ids = batch[f'input_ids_{idx}'].to(device)
            attention_mask = batch[f'attention_mask_{idx}'].to(device)
            
            if compute_logits:
                with torch.no_grad():
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                    topk_logits, topk_indices = torch.topk(logits, top_k, dim=-1)  # Get top-K logits and their indices

                # Convert the top-K logits and indices to lists and add to the respective lists
                topk_logits_list.extend(topk_logits.cpu().numpy().tolist())
                topk_indices_list.extend(topk_indices.cpu().numpy().tolist())

        if compute_logits:
            # Add the top-K logits and indices as new columns in the dataset
            tokenized_dataset = tokenized_dataset.add_column(f'topk_logits_model_{idx}', topk_logits_list)
            tokenized_dataset = tokenized_dataset.add_column(f'topk_indices_model_{idx}', topk_indices_list)

    return tokenized_dataset

@click.command()
@click.option("--k", type=int, default=128, help="Number of top logits to keep.")
@click.option("--dataset_names", type=str, default="p1atdev/ichikara-instruction:20231115-1,microsoft/orca-math-word-problems-200k,meta-math/MetaMathQA", help="Comma-separated list of dataset names to load.")
@click.option("--model_ids", type=str, default="augmxnt/shisa-gamma-7b-v1,WizardLM/WizardMath-7B-V1.1,arcee-train/Abel-7B-002-truncated-embeds", help="Comma-separated list of model IDs to load.")
@click.option("--base_model_name", type=str, default="mistralai/Mistral-7B-v0.1", help="Base model name.")
@click.option("--cache_dir", type=str, default="/home/ec2-user/.cache/huggingface", help="Cache directory.")
@click.option("--compute_logits", type=bool, default=True, help="Whether to compute logits or just keep input_ids.")
@click.option("--dataset_id", type=str, default="arcee-train/my-combined-dataset", help="Dataset ID to push to Hugging Face Hub.")
@click.option("--base_model_dataset_name", type=str, default="reflex-ai/fineweb-ultra-mini", help="Sample dataset name related to the base model.")
@click.option("--example_count", type=int, default=1729, help="Number of examples to select from each dataset.")
@click.option("--max_length", type=int, default=2048, help="Controls the length of the tokenized examples that will be used to train.")
@click.option("--add_top_k_logits", type=bool, default=False, help="Whether to add top-K logits to the combined dataset.")
@click.option("--seed", type=int, default=None, help="Seed used to shuffle the datasets")
def main(k, dataset_names, model_ids, base_model_name, cache_dir, compute_logits, dataset_id, base_model_dataset_name, example_count, max_length, add_top_k_logits, seed):
    # Environment variables
    
    # Setup tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True, cache_dir=cache_dir)
    
    # Convert comma-separated strings to lists
    dataset_names = dataset_names.split(',')
    model_ids = model_ids.split(',')
    
    # Load and template the datasets
    templated_datasets = setup_datasets_and_templates(tokenizer, dataset_names, example_count=example_count, seed=seed)

    # Data preprocessing
    combined_dataset, tokenizer = preprocess_data(templated_datasets=templated_datasets, 
                                                model_name=base_model_name, 
                                                cache_dir=cache_dir, 
                                                max_length=max_length,
                                                base_model_dataset_name=base_model_dataset_name)

    # Compute logits for each model and input set and save as additional columns if add_top_k_logits is True
    if add_top_k_logits:
        # Append base model name to model_ids if base_model_dataset_name is provided
        if base_model_dataset_name:
            model_ids.append(base_model_name)

        # Load additional models
        models_dict = load_models(model_ids, cache_dir)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        combined_dataset = compute_and_save_topk_logits(models_dict, combined_dataset, device, batch_size=4, top_k=k, compute_logits=compute_logits)

        # Save the dataset with logits to disk
        dataset_path = f"./DAM_logits_k_{k}"
        combined_dataset.save_to_disk(dataset_path)

        # Push the dataset to Hugging Face Hub
        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")

        try:
            # Load the dataset from the saved directory
            dataset = load_from_disk(dataset_path)
        except Exception as e:
            raise Exception(f"Error loading dataset from {dataset_path}: {e}")

        # Push the dataset to Hugging Face Hub
        dataset.push_to_hub(dataset_id)

        print("Dataset successfully uploaded to Hugging Face!")
    else:
        # Directly push the combined dataset to Hugging Face Hub
        combined_dataset.push_to_hub(dataset_id)
        print("Dataset successfully uploaded to Hugging Face without computing top-K logits!")

if __name__ == "__main__":
    main()