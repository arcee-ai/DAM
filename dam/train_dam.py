import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_from_disk, load_dataset
from model_preparation import prepare_model
from custom_trainer.dam_trainer import DAMTrainer  # Custom DAMTrainer
from transformers import TrainingArguments, default_data_collator
from modeling.dam import DAMBaseLayer
import click
import wandb
from pathlib import Path

# Environment variables
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

# Manual configurations
loss_fns = {
    "similarity": True, # default is True
    "l1_l2_reg": True, # default is True
    "overlap": False,   # default is False - proposed alternative to similarity
    "kl":True, # default is True
    "mse":False, # default is False - proposed alternative to kl
    "entropy":False # default is False - proposed alternative to kl
    }

# Command line arguments allow for WandB Sweep
@click.command()
@click.option("--temperature", default=2.0)
@click.option("--weight_decay", default=0.01)
@click.option("--learning_rate", default=1e-2)
@click.option("--lr_scheduler_type", default="linear")
@click.option("--lambda_coef_similarity", default=0.01)
@click.option("--lambda_coef_l1", default=1e-6)
@click.option("--lambda_coef_l2", default=1e-5)
@click.option("--use_wandb", default=True)
@click.option("--generate_logits_on_fly", default=True)
@click.option("--use_all_logits", default=True)
@click.option("--untrained_merged_model_name", default="arcee-train/pplist-merged-untrained-with-base-layernorm-embedding")
@click.option("--hf_disk_dataset_dir", default="arcee-train/logits-dataset-full-set-top-50")
@click.option("--cache_dir", default="/home/ec2-user/.cache/huggingface")
@click.option("--base_model_name", default="mistralai/Mistral-7B-v0.1")
def main(temperature, weight_decay, learning_rate, 
         lr_scheduler_type, lambda_coef_similarity, lambda_coef_l1, lambda_coef_l2,
         use_wandb, generate_logits_on_fly, use_all_logits,
         untrained_merged_model_name, hf_disk_dataset_dir, cache_dir, base_model_name):
    # Model and dataset details
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(untrained_merged_model_name, use_fast=True, cache_dir=cache_dir)

    # Load the dataset from disk
    # dataset = load_from_disk(hf_disk_dataset_dir)
    dataset = load_dataset(hf_disk_dataset_dir, split="train")

    # Prepare the model
    model = prepare_model(untrained_merged_model_name, cache_dir=cache_dir)

    print(f"The number of merged models is: {model.num_merged_models}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="no",
        save_strategy="no",
        do_eval=False,
        learning_rate=learning_rate,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        # max_steps=10,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        bf16=True,
        push_to_hub=False,
        remove_unused_columns=False,
        logging_dir='./logs',
        logging_steps=1,
        logging_strategy="steps",
        report_to="wandb" if use_wandb else "tensorboard",
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
    )

    # Initialize DAMTrainer
    trainer = DAMTrainer(
        model=model,  # Pass the main model here
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        lambda_coef_similarity=lambda_coef_similarity,  # Example lambda coefficient for regularization
        lambda_coef_l1=lambda_coef_l1,  # L1 regularization coefficient set to None
        lambda_coef_l2=lambda_coef_l2,  # L2 regularization coefficient
        temperature=temperature,  # Example temperature for KL divergence
        loss_fns=loss_fns,
        base_model_path=base_model_name,  # Pass base model as an argument
        generate_logits_on_fly=generate_logits_on_fly,
        use_all_logits=use_all_logits,
        use_wandb=use_wandb
    )

    if use_wandb:
        wandb.init(entity = 'arcee-ai', project="Dynamic Adaptive Merging")
        wandb.config.update(loss_fns)

    # Train the model
    trainer.train()

    # Save the trained model
    save_path = Path("saved_models") / wandb.run.name if use_wandb else Path("results") / "model"
    save_path.mkdir(parents=True, exist_ok=True)

    trainer.save_model(save_path)

    wandb.finish()

    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()

if __name__ == "__main__":
    main()