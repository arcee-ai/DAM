import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
from glom import glom, Assign
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import gc

from dam import DAMLayer
from utils import find_all_linear_names

os.environ['HF_TOKEN'] = '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'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
os.environ['HF_HOME'] = '/workspace/.hf'

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
# This list will be used later to apply the DAM (Dynamic Alignment Merger) layers
modules = find_all_linear_names(base_model)

truncated_layer = torch.nn.Linear(in_features=4096, out_features=32000, bias=False)

# Initializes the truncated_layer with a subset of the weights from model_C
with torch.no_grad():
    truncated_layer.weight.data = model_C.lm_head.weight.data[:32000, :]

assign = Assign('lm_head', truncated_layer)
# The glom function applies the assign operation to model_C, 
# replacing its lm_head with the truncated_layer. This effectively modifies model_C by substituting part of its original architecture with the new truncated layer.
glom(model_C, assign)

for m in tqdm(modules):
    # we access corresponding linear layers from all the models
    base_linear = glom(base_model, m)
    linear_a = glom(model_A, m)
    linear_b = glom(model_B, m)
    linear_c = glom(model_C, m)

    # This layer is designed to merge multiple linear layers (base_linear, linear_a, linear_b, and linear_c)
    # from different models during training, allowing the model to learn an optimal combination of these layers.
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

    # Replace the linear layer m in base_model with the newly created DAMLayer.
    assign = Assign(m, dam_layer)
    glom(base_model, assign)

# Define the DAM loss computation
def compute_dam_loss_for_lm(model, prompt_a, prompt_b, prompt_c, lambda_coef=0.01, temperature=2.0):
    batch = tokenizer([prompt_a, prompt_b, prompt_c], padding="max_length", truncation=True, max_length=4096, return_tensors='pt')

    input_ids = batch["input_ids"].to(model.device)
    attention_mask = batch["attention_mask"].to(model.device)

    input_a = input_ids[0].to(model.device)
    input_b = input_ids[1].to(model.device)
    input_c = input_ids[2].to(model.device)

    labels = batch["input_ids"].clone()
    labels[attention_mask == 0] = -100

    labels_a = labels[0].to(model.device)
    labels_b = labels[1].to(model.device)
    labels_c = labels[2].to(model.device)

    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, DAMLayer):
                module.set_forward_type('weight_1')
        original_logits_a = model(input_ids=input_a.unsqueeze(0), labels=labels_a.unsqueeze(0)).logits

        for module in model.modules():
            if isinstance(module, DAMLayer):
                module.set_forward_type('weight_2')
        original_logits_b = model(input_ids=input_b.unsqueeze(0), labels=labels_b.unsqueeze(0)).logits

        for module in model.modules():
            if isinstance(module, DAMLayer):
                module.set_forward_type('weight_3')
        original_logits_c = model(input_ids=input_c.unsqueeze(0), labels=labels_c.unsqueeze(0)).logits

    for module in model.modules():
        if isinstance(module, DAMLayer):
            module.set_forward_type('merge')   

    merged_logits_a = model(input_ids=input_a.unsqueeze(0), labels=labels_a.unsqueeze(0)).logits
    merged_logits_b = model(input_ids=input_b.unsqueeze(0), labels=labels_b.unsqueeze(0)).logits
    merged_logits_c = model(input_ids=input_c.unsqueeze(0), labels=labels_c.unsqueeze(0)).logits

    loss_a = F.kl_div(
        F.log_softmax(merged_logits_a / temperature, dim=-1),
        F.softmax(original_logits_a / temperature, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2) / len(input_a)
    
    loss_b = F.kl_div(
        F.log_softmax(merged_logits_b / temperature, dim=-1),
        F.softmax(original_logits_b / temperature, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2) / len(input_b)
    
    loss_c = F.kl_div(
        F.log_softmax(merged_logits_c / temperature, dim=-1),
        F.softmax(original_logits_c / temperature, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2) / len(input_c)
    
    similarity_loss = torch.tensor(0.0)
    for module in base_model.modules():
        if isinstance(module, DAMLayer):
            similarity_loss += module.compute_mergers_similarity().to(similarity_loss.device)
    
    total_loss = loss_a + loss_b + loss_c + lambda_coef * similarity_loss

    return total_loss

# Training loop
tokenizer.pad_token = tokenizer.eos_token
optimizer = torch.optim.AdamW(base_model.parameters(), lr=1e-3)
num_epochs = 1

plt.figure(figsize=(10,5))
plt.title("Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
losses = []

for epoch in range(num_epochs):
    base_model.train()
    total_loss = 0
    
    progress_bar = tqdm(zip(
        random.sample(templated_dataset_A['train']['text'], len(templated_dataset_A['train']['text'])),
        random.sample(templated_dataset_B['train']['text'], len(templated_dataset_B['train']['text'])),
        random.sample(templated_dataset_C['train']['text'], len(templated_dataset_C['train']['text']))
    ), 
    total=min([len(templated_dataset_A['train']['text']), len(templated_dataset_B['train']['text']), len(templated_dataset_C['train']['text'])]),
    desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for i, (prompt_a, prompt_b, prompt_c) in enumerate(progress_bar):
        loss = compute_dam_loss_for_lm(base_model, prompt_a, prompt_b, prompt_c)

        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        current_loss = loss.item()
        total_loss += current_loss
        losses.append(current_loss)

        progress_bar.set_postfix({'loss': f'{current_loss:.4f}'})

        if i % 10 == 0:
            plt.clf()
            plt.plot(losses)
            plt.title("Training Loss")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.show()
    
    avg_loss = total_loss / len(progress_bar)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

plt.figure(figsize=(10,5))
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

# Final model merging
new_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
for (name, module), (_, new_module) in tqdm(zip(base_model.named_modules(), new_model.named_modules()), 
                                            desc="Merging layers"):
    if isinstance(module, DAMLayer):
        merged_weight = module.get_dam_weight()
        merged_bias = module.get_dam_bias()
        
        new_module.weight.data = merged_weight
        if merged_bias is not None:
            new_module.bias.data = merged_bias

# Save the merged model
new_model.save_pretrained("/workspace/zip_merged_della_delta3")
tokenizer.save_pretrained("/workspace/zip_merged_della_delta3")

print("Merged model saved successfully!")

# Cleanup
model_A.cpu()
model_B.cpu()
model_C.cpu()
del model_A, model_B, model_C
gc.collect()
torch.cuda.empty_cache()
