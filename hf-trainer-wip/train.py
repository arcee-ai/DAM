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

from dam import DAMLinearLayer, DAMEmbeddingLayer, MergedModel  # Updated import for both DAM layers
from utils import find_all_linear_names, find_embedding_layers  # Updated import for finding embedding layers

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

tokenizer.chat_template = "{%- set ns = namespace(found=false) -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {%- set ns.found = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- if not ns.found -%}\n    {{- '' + 'Below is an instruction that describes a task. Write a response that appropriately completes the request.' + '\\n\\n' -}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' -%}\n        {{- '' + message['content'] + '\\n\\n' -}}\n    {%- else -%}\n        {%- if message['role'] == 'user' -%}\n            {{-'### Instruction:\\n' + message['content'] + '\\n\\n'-}}\n        {%- else -%}\n            {{-'### Response:\\n' + message['content'] + '\\n\\n' -}}\n        {%- endif -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{-'### Response:\\n'-}}\n{%- endif -%}"

# Load and template datasets
dataset_A = load_dataset("p1atdev/ichikara-instruction", '20231115-1').rename_column("text", "instruction")
dataset_B = load_dataset("microsoft/orca-math-word-problems-200k")
dataset_C = load_dataset("meta-math/MetaMathQA")

templated_dataset_A = dataset_A.map(lambda row: {'text': tokenizer.apply_chat_template(
    [{'role': 'user', 'content': row['instruction']}, {'role': 'assistant', 'content': row['output']}], 
    tokenize=False).strip()})
templated_dataset_B = dataset_B.map(lambda row: {'text': tokenizer.apply_chat_template(
    [{'role': 'user', 'content': row['question']}, {'role': 'assistant', 'content': row['answer']}], 
    tokenize=False).strip()})
templated_dataset_C = dataset_C.map(lambda row: {'text': tokenizer.apply_chat_template(
    [{'role': 'user', 'content': row['query']}, {'role': 'assistant', 'content': row['response']}], 
    tokenize=False).strip()})

# Prepare DAM layers
linear_modules = find_all_linear_names(base_model)
embedding_modules = find_embedding_layers(base_model)

# Truncate model_C's layers to match the base model's vocabulary size
truncated_lm_head = torch.nn.Linear(in_features=4096, out_features=32000, bias=False)
truncated_embed = torch.nn.Embedding(num_embeddings=32000, embedding_dim=4096)

with torch.no_grad():
    truncated_lm_head.weight.data = model_C.lm_head.weight.data[:32000, :]
    truncated_embed.weight.data = model_C.model.embed_tokens.weight.data[:32000, :]

# Apply the truncation to the embedding and linear layers of model_C
assign_embed = Assign('model.embed_tokens', truncated_embed)
assign_lm_head = Assign('lm_head', truncated_lm_head)
glom(model_C, assign_embed)
glom(model_C, assign_lm_head)

# Loop over both linear and embedding layers to replace them with DAM layers
for m in tqdm(linear_modules + embedding_modules):
    module_a = glom(model_A, m)
    module_b = glom(model_B, m)
    module_c = glom(model_C, m)

    dam_layer = None
    
    if isinstance(module_a, torch.nn.Embedding):
        dam_layer = DAMEmbeddingLayer(
            embedding_a=module_a,
            embedding_b=module_b,
            embedding_c=module_c,
            device=base_model.device,
            dtype=module_a.weight.dtype
        )
    else:
        dam_layer = DAMLinearLayer(
            base_linear=glom(base_model, m),
            linear_a=module_a,
            linear_b=module_b,
            linear_c=module_c,
            device=base_model.device,
            dtype=module_a.weight.dtype
        )

    assign = Assign(m, dam_layer)
    glom(base_model, assign)

# Define the DAM loss computation
def compute_dam_loss_for_lm(model, prompt_a, prompt_b, prompt_c, lambda_coef=0.01, temperature=2.0, lambda_coef_reg=None):
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
            if isinstance(module, (DAMLinearLayer, DAMEmbeddingLayer)):
                module.set_forward_type('weight_1')
        original_logits_a = model(input_ids=input_a.unsqueeze(0), labels=labels_a.unsqueeze(0)).logits

        for module in model.modules():
            if isinstance(module, (DAMLinearLayer, DAMEmbeddingLayer)):
                module.set_forward_type('weight_2')
        original_logits_b = model(input_ids=input_b.unsqueeze(0), labels=labels_b.unsqueeze(0)).logits

        for module in model.modules():
            if isinstance(module, (DAMLinearLayer, DAMEmbeddingLayer)):
                module.set_forward_type('weight_3')
        original_logits_c = model(input_ids=input_c.unsqueeze(0), labels=labels_c.unsqueeze(0)).logits
    

    for module in model.modules():
        if isinstance(module, (DAMLinearLayer, DAMEmbeddingLayer)):
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
    similarity_reg_loss = torch.tensor(0.0)
    for module in model.modules():
        if isinstance(module, (DAMLinearLayer, DAMEmbeddingLayer)):
            similarity_reg_loss += module.compute_mergers_similarity(lambda_coef_reg=lambda_coef_reg).to(similarity_reg_loss.device)

    total_loss = loss_a + loss_b + loss_c + lambda_coef * similarity_loss + similarity_reg_loss

    return total_loss


# Freeze all parameters
for param in base_model.parameters():
    param.requires_grad = False

# Unfreeze merging coefficients in DAMLinearLayer and DAMEmbeddingLayer
for module in base_model.modules():
    if isinstance(module, (DAMLinearLayer, DAMEmbeddingLayer)):
        # Unfreeze merging coefficients
        module.merger_1.requires_grad = True
        module.merger_2.requires_grad = True
        module.merger_3.requires_grad = True
        
        if hasattr(module, 'bias_merger1') and module.bias_merger1 is not None:
            module.bias_merger1.requires_grad = True
        if hasattr(module, 'bias_merger2') and module.bias_merger2 is not None:
            module.bias_merger2.requires_grad = True
        if hasattr(module, 'bias_merger3') and module.bias_merger3 is not None:
            module.bias_merger3.requires_grad = True

# Create an optimizer that only updates the unfreezing parameters
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, base_model.parameters()), lr=1e-3)

# Training loop
tokenizer.pad_token = tokenizer.eos_token
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
        loss = compute_dam_loss_for_lm(base_model, prompt_a, prompt_b, prompt_c, lambda_coef=0.01, lambda_coef_reg=0.01)

        print(loss)
        exit()

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


# Final model merging
new_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
for (name, module), (_, new_module) in tqdm(zip(base_model.named_modules(), new_model.named_modules()), 
                                            desc="Merging layers"):
    if isinstance(module, (DAMLinearLayer, DAMEmbeddingLayer)):
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
