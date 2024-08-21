import os
from transformers import AutoTokenizer
from data_setup import setup_datasets_and_templates
from data_preprocessing import preprocess_data
from model_preparation import prepare_model
from dam_trainer import DAMTrainer  # Custom DAMTrainer
from transformers import TrainingArguments, default_data_collator

# Environment variables
os.environ['HF_TOKEN'] = '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'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
os.environ['HF_HOME'] = '/workspace/.hf'

def main():
    # Model and dataset details
    model_name = "arcee-ai/untrained-DAM-merge-01"
    cache_dir = "/workspace/.hf"

    # Setup tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=cache_dir)
    templated_datasets = setup_datasets_and_templates(tokenizer, example_count=1069)

    # Data preprocessing
    combined_dataset, tokenizer = preprocess_data(templated_datasets, model_name, cache_dir)
    model = prepare_model(model_name, cache_dir)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="no",
        save_strategy="no", 
        do_eval=False,     
        learning_rate=1e-3,
        per_device_train_batch_size=1,
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
    )

    # Initialize DAMTrainer
    trainer = DAMTrainer(
        model=model,
        args=training_args,
        train_dataset=combined_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    # Train the model
    trainer.train()

    trainer.save_model()

if __name__ == "__main__":
    main()
