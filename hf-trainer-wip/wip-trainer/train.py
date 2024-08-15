import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, HfArgumentParser
from datasets import load_dataset
from dam import DAMLayer
from utils import find_all_linear_names
from custom_trainer import CustomTrainer  # Assuming you save CustomTrainer in a file named custom_trainer.py

# Environment Setup
os.environ['HF_TOKEN'] = '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'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

# Model IDs
MODEL_ID = "mistralai/Mistral-7B-v0.1"
MODEL_ID_A = "augmxnt/shisa-gamma-7b-v1"
MODEL_ID_B = "WizardLM/WizardMath-7B-V1.1"
MODEL_ID_C = "GAIR/Abel-7B-002"

# Load models and tokenizer
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
model_A = AutoModelForCausalLM.from_pretrained(MODEL_ID_A, torch_dtype=torch.bfloat16, device_map="auto")
model_B = AutoModelForCausalLM.from_pretrained(MODEL_ID_B, torch_dtype=torch.bfloat16, device_map="auto")
model_C = AutoModelForCausalLM.from_pretrained(MODEL_ID_C, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

# Prepare DAM layers
modules = find_all_linear_names(base_model)

for m in modules:
    base_linear = glom(base_model, m)
    linear_a = glom(model_A, m)
    linear_b = glom(model_B, m)
    linear_c = glom(model_C, m)

    dam_layer = DAMLayer(
        base_linear=base_linear,
        linear_a=linear_a,
        linear_b=linear_b,
        linear_c=linear_c,
        init_merger_value=0.0,
        init_merger_value_2=0.0,
        init_merger_value_3=0.0,
        device=base_model.device,
        dtype=linear_a.weight.dtype
    )

    assign = Assign(m, dam_layer)
    glom(base_model, assign)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
)

# Load dataset
templated_dataset_A = load_dataset("path/to/dataset_A")
templated_dataset_B = load_dataset("path/to/dataset_B")
templated_dataset_C = load_dataset("path/to/dataset_C")

# Initialize Trainer
trainer = CustomTrainer(
    model=base_model,
    args=training_args,
    train_dataset=templated_dataset_A["train"],  # Replace with appropriate dataset
    eval_dataset=templated_dataset_A["validation"],  # Replace with appropriate dataset
    tokenizer=tokenizer,
)

# Training
trainer.train()

# Save the merged model
new_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
for (name, module), (_, new_module) in zip(base_model.named_modules(), new_model.named_modules()):
    if isinstance(module, DAMLayer):
        merged_weight = module.get_dam_weight()
        merged_bias = module.get_dam_bias()
        
        new_module.weight.data = merged_weight
        if merged_bias is not None:
            new_module.bias.data = merged_bias

new_model.save_pretrained("/workspace/zip_merged_della_delta3")
tokenizer.save_pretrained("/workspace/zip_merged_della_delta3")

print("Merged model saved successfully!")

# Cleanup
model_A.cpu()
model_B.cpu()
model_C.cpu()
del model_A, model_B, model_C
torch.cuda.empty_cache()
