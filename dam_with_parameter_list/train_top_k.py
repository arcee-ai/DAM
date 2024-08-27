import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_from_disk, load_dataset
from model_preparation import prepare_model
from custom_trainer.dam_trainer_top_k import DAMTrainer  # Custom DAMTrainer
from transformers import TrainingArguments, default_data_collator
from modeling.dam import DAMBaseLayer


# Environment variables
os.environ['HF_TOKEN'] = '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' #'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'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
os.environ['HF_HOME'] = '/workspace/hf-cache'

def main():
    # Model and dataset details
    base_model_name = "mistralai/Mistral-7B-v0.1" 
    model_name = "arcee-train/pplist-merged-untrained"
    cache_dir = "/workspace/hf-cache"
    hf_disk_dataset_dir = "arcee-train/logits-dataset-mock"

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True, cache_dir=cache_dir)

    # Load the dataset from disk
    #dataset = load_from_disk(hf_disk_dataset_dir)
    dataset = load_dataset(hf_disk_dataset_dir, split="train")

    # Prepare the model
    model = prepare_model(model_name, cache_dir)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="no",
        save_strategy="no",
        do_eval=False,
        learning_rate=1e-3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        weight_decay=0.00,
        lr_scheduler_type='constant',
        bf16=True,
        push_to_hub=False,
        remove_unused_columns=False,
        logging_dir='./logs',
        logging_steps=1,
        logging_strategy="steps",
        report_to="tensorboard",
        gradient_accumulation_steps=1
    )

    
    # Initialize DAMTrainer
    trainer = DAMTrainer(
        model=model,  # Pass the main model here
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        lambda_coef=0.01,  # Example lambda coefficient for regularization
        lambda_coef_reg=0.0001,  # Example lambda coefficient for regularization
        temperature=2.0,  # Example temperature for KL divergence
        use_kl=False,
        use_mse=True
    )

    # Train the model
    trainer.train()

    # Save the trained model
    trainer.save_model()

if __name__ == "__main__":
    main()