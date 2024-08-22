import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from data_setup import setup_datasets_and_templates
from data_preprocessing import preprocess_data
from model_preparation import prepare_model
from dam_trainer import DAMTrainer  # Custom DAMTrainer
from transformers import TrainingArguments, default_data_collator
from modeling.dam import DAMBaseLayer

# Environment variables
os.environ['HF_TOKEN'] = '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'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
os.environ['HF_HOME'] = '/workspace/hf-cache'

def print_trainable_params(model, title="Trainable Parameters"):
    print(f"\n{title}:")
    trainable_params_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            #print(f" - {name}: {param.shape}")
            trainable_params_count += param.numel()
    print(f"Total trainable parameters: {trainable_params_count}")

def main():
    # Model and dataset details
    model_name = "arcee-ai/untrained-DAM-merge-01"
    cache_dir = "/workspace/hf-cache"

    # Setup tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True, cache_dir=cache_dir)
    templated_datasets = setup_datasets_and_templates(tokenizer, example_count=1069)

    # Data preprocessing
    combined_dataset, tokenizer = preprocess_data(templated_datasets, "mistralai/Mistral-7B-v0.1", cache_dir)
    
    # Load additional models
    MODEL_ID_A = "augmxnt/shisa-gamma-7b-v1"
    MODEL_ID_B = "WizardLM/WizardMath-7B-V1.1"
    MODEL_ID_C = "arcee-train/Abel-7B-002-truncated-embeds"

    model = prepare_model(model_name, cache_dir)

    print_trainable_params(model, title="Before Freezing")

    # freeze the base model (WE DO NOT NEED TO DO THIS SINCE WE FREEZE THE BASE MODEL IN THE DAMBaseLayer)
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze the merging coefficients
    for module in model.modules():
        if isinstance(module, DAMBaseLayer):  # Identify your custom layers
            for merger in module.mergers:
                merger.requires_grad = True  # Unfreeze only the merging coefficients
            if hasattr(module, 'bias_mergers'):  # Check if there are bias mergers
                for bias_merger in module.bias_mergers:
                    bias_merger.requires_grad = True

    print_trainable_params(model, title="After Freezing")

    # model_A = AutoModelForCausalLM.from_pretrained(MODEL_ID_A, torch_dtype=torch.bfloat16, device_map="auto", cache_dir=cache_dir)
    # model_B = AutoModelForCausalLM.from_pretrained(MODEL_ID_B, torch_dtype=torch.bfloat16, device_map="auto", cache_dir=cache_dir)
    # model_C = AutoModelForCausalLM.from_pretrained(MODEL_ID_C, torch_dtype=torch.bfloat16, device_map="auto", cache_dir=cache_dir)

    model_A = AutoModelForCausalLM.from_pretrained(MODEL_ID_A, torch_dtype=torch.bfloat16,  cache_dir=cache_dir)
    model_B = AutoModelForCausalLM.from_pretrained(MODEL_ID_B, torch_dtype=torch.bfloat16,  cache_dir=cache_dir)
    model_C = AutoModelForCausalLM.from_pretrained(MODEL_ID_C, torch_dtype=torch.bfloat16,  cache_dir=cache_dir)

    # Dictionary of models
    models_dict = {
        "model_1": model_A,
        "model_2": model_B,
        "model_3": model_C
    }

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
        model=model,  # Pass the main model here
        models_dict=models_dict,
        args=training_args,
        train_dataset=combined_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        lambda_coef=0.01,  # Example lambda coefficient for regularization
        temperature=2.0  # Example temperature for KL divergence
    )

    # Train the model
    trainer.train()

    # Save the trained model
    trainer.save_model()

if __name__ == "__main__":
    main()
