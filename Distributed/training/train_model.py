import torch
from transformers import Trainer, TrainingArguments
from datasets import load_from_disk
from merged_model import MergedModel
from transformers import AutoTokenizer

def train_model(dataset_path, model_name, output_dir="output"):
    combined_dataset = load_from_disk(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    model = MergedModel.from_pretrained("merged_model")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=combined_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

if __name__ == "__main__":
    train_model("combined_dataset_with_logits", "mistralai/Mistral-7B-v0.1")
