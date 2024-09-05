from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


def preprocess_data(templated_datasets, model_name, cache_dir=None, max_length=2048):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=cache_dir)

    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token

    def tokenize_and_align(examples):
        tokenized = tokenizer(
            examples['text'], 
            padding="max_length", 
            truncation=True, 
            max_length=max_length, 
            return_tensors="np"  # Change to 'np' to keep as numpy arrays, which are easier to manipulate
        )
        return tokenized

    # Tokenize and align each dataset
    tokenized_datasets = [ds.map(tokenize_and_align, batched=True, remove_columns=['text']) for ds in templated_datasets]

    # Combine datasets into one dataset with multiple input_ids and attention_mask columns
    combined_examples = []
    for rows in zip(*tokenized_datasets):
        combined_row = {}
        for idx, row in enumerate(rows):
            input_ids_key = f'input_ids_{idx+1}'
            attention_mask_key = f'attention_mask_{idx+1}'
            labels_key = f'labels_{idx+1}'
            
            combined_row[input_ids_key] = row['input_ids']
            combined_row[attention_mask_key] = row['attention_mask']
            
            # Create labels where padding tokens are set to -100
            labels = [
                token_id if mask == 1 else -100
                for token_id, mask in zip(row['input_ids'], row['attention_mask'])
            ]
            combined_row[labels_key] = labels

        combined_examples.append(combined_row)

    # Convert the combined examples into a new Dataset
    combined_dataset = Dataset.from_dict({key: [row[key] for row in combined_examples] for key in combined_examples[0].keys()})

    return combined_dataset, tokenizer
