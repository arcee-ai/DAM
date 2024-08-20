import os
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

def prepare_datasets(model_names, example_count=None, max_length=512):
    tokenizer = AutoTokenizer.from_pretrained(model_names[0], use_fast=True)
    
    def get_logit_for_example(model, tokenizer, text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        return logits

    # Load and process each dataset
    datasets = [load_dataset(name) for name in model_names]
    processed_datasets = []

    for model_name, dataset in zip(model_names, datasets):
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if example_count:
            dataset = dataset["train"].select(range(example_count))
        
        def process_row(row):
            logits = get_logit_for_example(model, tokenizer, row['text'])
            return {'logits': logits.cpu().numpy()}
        
        processed_dataset = dataset.map(process_row, batched=False)
        processed_datasets.append(processed_dataset)

    # Combine processed datasets into a single dataset with logits from all models
    combined_dataset = Dataset.from_dict({
        f"logits_{i}": processed_datasets[i]['logits'] for i in range(len(processed_datasets))
    })

    return combined_dataset

if __name__ == "__main__":
    model_names = [
        "mistralai/Mistral-7B-v0.1",
        "augmxnt/shisa-gamma-7b-v1",
        "WizardLM/WizardMath-7B-V1.1",
        "GAIR/Abel-7B-002"
    ]
    combined_dataset = prepare_datasets(model_names, example_count=10)
    combined_dataset.save_to_disk("combined_dataset_with_logits")
