import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_from_disk, load_dataset
from model_preparation import prepare_model
from custom_trainer.dam_trainer import DAMTrainer
from transformers import TrainingArguments, default_data_collator
from modeling.dam import DAMBaseLayer
import click
import wandb
from pathlib import Path
from optimizer import MoFOAdamW

# Environment variables
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

# Command line arguments
@click.command()
@click.option("--temperature", default=2.0, help="Temperature for KL divergence.")
@click.option("--weight_decay", default=0.0, help="Weight decay for the optimizer.")
@click.option("--learning_rate", default=1e-3, help="Learning rate for the optimizer.")
@click.option("--lr_scheduler_type", default="constant", help="Type of learning rate scheduler (`linear`, etc.).")
@click.option("--warmup_ratio", default=0.0, help="Warmup ratio for learning rate scheduler.")
@click.option("--mofo_alpha", default=0.3, help="The alpha parameter for the MoFO optimizer (fraction of gradients kept based on momentum). Set to None to use AdamW")
@click.option("--lambda_coef_similarity", default=0.01, help="Lambda coefficient for similarity regularization.")
@click.option("--lambda_coef_l1", default=0.0, help="L1 regularization coefficient.")
@click.option("--lambda_coef_l2", default=0.0, help="L2 regularization coefficient.")
@click.option("--per_device_train_batch_size", default=1, help="Per device train batch size.")
@click.option("--gradient_accumulation_steps", default=1, help="Number of gradient accumulation steps.")
@click.option("--use_wandb", type=click.BOOL, default=True, help="Upload training logs to Weights and Biases.")
@click.option("--generate_logits_on_fly", type=click.BOOL, default=True, help="Generate logits on-the-fly during training.")
@click.option("--use_all_logits", type=click.BOOL, default=True, help="Use all logits during training.")
@click.option("--untrained_merged_model_name", default="arcee-train/pplist-merged-untrained-linear-only-no-base", help="Name of the untrained merged model.")
@click.option("--combined_hf_dataset_dir", default="arcee-train/DAM_combined_no_base_no_logits", help="Directory of the dataset with logits.")
@click.option("--cache_dir", default="/home/ec2-user/.cache/huggingface", help="Directory to cache the models.")
@click.option("--base_model_name", default="mistralai/Mistral-7B-v0.1", help="Name of the base model.")
@click.option("--similarity", type=click.BOOL, default=True, help="Use similarity loss.")
@click.option("--l1_l2_reg", type=click.BOOL, default=False, help="Use L1/L2 regularization.")
@click.option("--overlap", type=click.BOOL, default=False, help="Use overlap loss.")
@click.option("--kl", type=click.BOOL, default=True, help="Use KL divergence loss.")
@click.option("--mse", type=click.BOOL, default=False, help="Use MSE loss.")
@click.option("--entropy", type=click.BOOL, default=False, help="Use entropy loss.")
def main(temperature, weight_decay, learning_rate, 
         lr_scheduler_type, warmup_ratio, mofo_alpha, lambda_coef_similarity, lambda_coef_l1, lambda_coef_l2,
         per_device_train_batch_size, gradient_accumulation_steps, use_wandb, generate_logits_on_fly, use_all_logits,
         untrained_merged_model_name, combined_hf_dataset_dir, cache_dir, base_model_name,
         similarity, l1_l2_reg, overlap, kl, mse, entropy):
    cmd_args = locals().copy()

    # Model and dataset details
    # Prepare the model
    model = prepare_model(untrained_merged_model_name, cache_dir=cache_dir)
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(untrained_merged_model_name, use_fast=True, cache_dir=cache_dir)

    # Load the dataset
    # dataset = load_from_disk(combined_hf_dataset_dir)
    dataset = load_dataset(combined_hf_dataset_dir, split="train")

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="no",
        save_strategy="no",
        do_eval=False,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        bf16=True,
        push_to_hub=False,
        remove_unused_columns=False,
        logging_dir='./logs',
        logging_steps=1,
        logging_strategy="steps",
        report_to="wandb" if use_wandb else "tensorboard",
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=1.0,
    )

    optimizer = MoFOAdamW(model.parameters(), lr=learning_rate, alpha=mofo_alpha) if mofo_alpha else None
    loss_fns = {
        "similarity": similarity, 
        "l1_l2_reg": l1_l2_reg, 
        "overlap": overlap,  
        "kl": kl, 
        "mse": mse, 
        "entropy": entropy 
        }
    
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
        use_wandb=use_wandb,
        optimizers=(optimizer, None)
    )

    if use_wandb:
        wandb.init(entity = 'arcee-ai', project="Dynamic Adaptive Merging")
        wandb.config.update(loss_fns)
        # Get the keys from TrainingArguments to exclude them from cmd_args
        training_args_keys = set(training_args.to_dict().keys())

        # Create a dictionary of command-line arguments to log
        cmd_args_to_log = {k: v for k, v in cmd_args.items() if k not in training_args_keys}
        wandb.config.update(cmd_args_to_log)


    # Train the model
    trainer.train()

    # Save the trained model
    save_path = Path(__file__).parent / "saved_models"
    if use_wandb:
        save_path /= wandb.run.name
    else:
        save_path /= "model"
    save_path.mkdir(parents=True, exist_ok=True)

    trainer.save_model(save_path)

    if use_wandb:
        wandb.finish()

    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()

if __name__ == "__main__":
    main()

# python dam/train_dam.py --temperature 2.0 --weight_decay 0.0 --learning_rate 1e-2 --lr_scheduler_type linear --warmup_ratio 0.1 --lambda_coef_similarity 0.01 --lambda_coef_l1 0.0 --lambda_coef_l2 0.0 --generate_logits_on_fly True --use_all_logits True --untrained_merged_model_name /home/ec2-user/shamane/DAM/merged_model  --combined_hf_dataset_dir arcee-train/my-combined-dataset --cache_dir /home/ec2-user/.cache/huggingface --base_model_name mistralai/Mistral-7B-v0.1 --use_wandb True