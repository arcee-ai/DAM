import os
import torch
import torch.nn.functional as F
from transformers import Trainer, AutoModelForCausalLM
from modeling.dam import DAMLinearLayer, DAMEmbeddingLayer, DAMRMSNorm
from safetensors.torch import save_file
from tqdm import tqdm
from glom import glom

try:
    import wandb
except:
    pass

def set_model_index(model, index):
    for module in model.modules():
        if isinstance(module, (DAMLinearLayer, DAMEmbeddingLayer, DAMRMSNorm, torch.nn.Linear)):
            module.model_index = index

def kl_divergence_loss(logits, target_logits, non_padded_tokens, temperature=1.0):
    # Compute the KL divergence between the log-softmax of logits and the softmax of target_logits
    kl_div = F.kl_div(
        F.log_softmax(logits / temperature, dim=-1),
        F.softmax(target_logits / temperature, dim=-1),
        reduction='batchmean'
    )
    
    # Scale the KL divergence by the temperature squared
    scaled_kl_div = kl_div * (temperature ** 2)
    
    # Normalize by the number of non-padded tokens
    normalized_kl_div = scaled_kl_div / non_padded_tokens
    
    return normalized_kl_div

def entropy_loss(logits, attention_mask, non_padded_tokens, temperature=1.0, lambda_coef_entropy=0.01):
    # Apply softmax to the logits
    probabilities = F.softmax(logits / temperature, dim=-1)
    
    # Compute the entropy
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=-1)
    
    # Mask out the padding tokens in the entropy
    masked_entropy = entropy * attention_mask
    
    # Compute the total entropy loss, considering only non-padded tokens
    total_entropy_loss = masked_entropy.sum() * (temperature ** 2)
    
    # Normalize by the number of non-padded tokens
    return lambda_coef_entropy * total_entropy_loss / non_padded_tokens

def mse_loss(logits, target_logits, non_padded_tokens, lambda_coef_mse=0.001):
    # Compute the MSE loss
    mse_loss = F.mse_loss(logits, target_logits, reduction='none').mean(dim=-1).sum()
    
    # Normalize by the number of non-padded tokens
    return lambda_coef_mse * mse_loss / non_padded_tokens

class DAMTrainer(Trainer):
    def __init__(self, model, 
                 lambda_coef_similarity=0.01,  # Added similarity regularization coefficient
                 lambda_coef_l1=1e-6,  # Added L1 regularization coefficient
                 lambda_coef_l2=1e-6,  # Added L2 regularization coefficient
                 lambda_coef_overlap=1e-5,  # Added overlap regularization coefficient
                 lambda_coef_mse=1.0,  # Added MSE regularization coefficient
                 lambda_coef_entropy=0.1,  # Added entropy regularization coefficient
                 temperature=2.0, 
                 loss_fns=None,
                 base_model_path=None, 
                 use_wandb=True,
                 generate_logits_on_fly=False,  # New parameter to control logits generation
                 use_all_logits=False,
                 report_all_metrics=False,
                 **kwargs):
        super().__init__(model=model, **kwargs)
        self.lambda_coef_similarity = lambda_coef_similarity
        self.lambda_coef_l1 = lambda_coef_l1  # Initialize L1 regularization coefficient
        self.lambda_coef_l2 = lambda_coef_l2  # Initialize L2 regularization coefficient
        self.lambda_coef_overlap = lambda_coef_overlap
        self.lambda_coef_mse = lambda_coef_mse
        self.lambda_coef_entropy = lambda_coef_entropy
        self.temperature = temperature
        self.use_all_logits = use_all_logits

        self.loss_fns = loss_fns

        self.base_model_path = base_model_path
        self.use_wandb = use_wandb
        self.generate_logits_on_fly = generate_logits_on_fly 

        self.report_all_metrics = report_all_metrics

        assert not (self.use_all_logits and not self.generate_logits_on_fly), "You can't have use_all_logits=True if generate_logits_on_fly is False"

    def compute_individual_logit_losses(self, merged_logits, individual_logits, attention_mask, non_padded_tokens, dataple_id):
        masked_merged_logits = merged_logits * attention_mask.unsqueeze(-1)
        masked_individual_logits = individual_logits * attention_mask.unsqueeze(-1)

        total_loss = 0.0
        loss_logs = {}

        if self.loss_fns['kl'] or self.report_all_metrics:
            kl_loss = kl_divergence_loss(masked_merged_logits, 
                                            masked_individual_logits, 
                                            non_padded_tokens, 
                                            temperature=self.temperature)
            loss_logs[f'kl_loss_{dataple_id}'] = kl_loss
            if self.loss_fns['kl']:
                total_loss += kl_loss

        if self.loss_fns['mse'] or self.report_all_metrics:
            mse_loss_value = mse_loss(masked_merged_logits,
                                        masked_individual_logits, 
                                        non_padded_tokens,
                                        lambda_coef_mse=self.lambda_coef_mse)
            loss_logs[f'mse_loss_{dataple_id}'] = mse_loss_value
            if self.loss_fns['mse']:
                total_loss += mse_loss_value

        if self.loss_fns['entropy'] or self.report_all_metrics:
            e_loss = entropy_loss(masked_merged_logits, 
                                    attention_mask, 
                                    non_padded_tokens,
                                    temperature=self.temperature,
                                    lambda_coef_entropy=self.lambda_coef_entropy)
            loss_logs[f'entropy_loss_{dataple_id}'] = e_loss
            if self.loss_fns['entropy']:
                total_loss += e_loss

        return total_loss, loss_logs

    def compute_loss(self, merged_model, inputs, return_outputs=False, num_items_in_batch=None):
        # Ensure the merged_model is on the correct device
        device = merged_model.device
        
        # Dynamically identify the number of input types (e.g., input_ids_1, input_ids_2, etc.)
        input_keys = [key for key in inputs.keys() if key.startswith('input_ids_')]
        num_datasets = len(input_keys)

        # Prepare dictionaries to store input_ids and attention_masks
        input_ids_dict = {}
        attention_mask_dict = {}
        
        for dataset_idx in range(1, num_datasets + 1):
            input_ids_dict[f'input_ids_{dataset_idx}'] = inputs[f'input_ids_{dataset_idx}'].to(device)
            attention_mask_dict[f'attention_mask_{dataset_idx}'] = inputs[f'attention_mask_{dataset_idx}'].to(device)

        # Compute logits for the merged model separately for each input
        merged_logits_dict = {}
        for dataset_idx in range(1, num_datasets + 1):
            input_ids = input_ids_dict[f'input_ids_{dataset_idx}']
            attention_mask = attention_mask_dict[f'attention_mask_{dataset_idx}']

            merged_logits = merged_model(input_ids=input_ids, attention_mask=attention_mask).logits
            merged_logits_dict[f'merged_logits_{dataset_idx}'] = merged_logits

        # Compute logits for each individual model with the respective dataset if generate_logits_on_fly is True
        if self.generate_logits_on_fly:
            individual_logits_dict = {}
            
            # we do not compute logits for the base model
            for model_index in range(num_datasets):
                # set the model index for the merged model to get correct logits from individual models.
                set_model_index(merged_model, index=model_index)
                input_ids = input_ids_dict[f'input_ids_{model_index + 1}']
                attention_mask = attention_mask_dict[f'attention_mask_{model_index + 1}']
                with torch.no_grad():
                    # device = merged_model.get_input_embeddings().device
                    model_logits = merged_model(input_ids=input_ids, attention_mask=attention_mask).logits
                individual_logits_dict[model_index] = model_logits.to(device)
            
            # Reset model_index to None after computing logits for individual models
            set_model_index(merged_model, index=None)  # Reset model_index to None for merged model computations
   
    
        total_loss = 0.0
        all_loss_logs = {}

        # Compute KL divergence loss between the merged model's logits and each corresponding top-K logits
        for dataset_idx in range(1, num_datasets + 1):
            merged_logits = merged_logits_dict[f'merged_logits_{dataset_idx}']
            attention_mask = attention_mask_dict[f'attention_mask_{dataset_idx}']
            non_padded_tokens = attention_mask.sum().item()

            if self.generate_logits_on_fly:
                individual_logits = individual_logits_dict[dataset_idx - 1]
            else:
                individual_logits = inputs[f'topk_logits_model_{dataset_idx}'].to(device)
                topk_indices = inputs[f'topk_indices_model_{dataset_idx}'].to(device)
                merged_logits = torch.gather(merged_logits, dim=-1, index=topk_indices)
            
            loss, loss_logs = self.compute_individual_logit_losses(merged_logits, 
                                                                   individual_logits, 
                                                                   attention_mask, 
                                                                   non_padded_tokens, 
                                                                   dataset_idx)
            total_loss += loss
            all_loss_logs.update(loss_logs)

        # Compute similarity loss and L2 regularization for merging coefficients
        similarity_loss = torch.tensor(0.0, device=device)
        l1_l2_reg = torch.tensor(0.0, device=device)
        overlap_loss = torch.tensor(0.0, device=device)
        weighted_overlap_loss = torch.tensor(0.0, device=device)
        for module in merged_model.modules():
            if hasattr(module, 'compute_mergers_similarity') and (self.loss_fns['similarity'] or self.report_all_metrics):
                similarity_loss += module.compute_mergers_similarity(self.lambda_coef_similarity).to(similarity_loss.device)
            if hasattr(module, 'compute_mergers_L1_L2_reg') and (self.loss_fns['l1_l2_reg'] or self.report_all_metrics):
                l1_l2_reg += module.compute_mergers_L1_L2_reg(
                    lambda_coef_l1=self.lambda_coef_l1, 
                    lambda_coef_l2=self.lambda_coef_l2
                ).to(l1_l2_reg.device)
            if hasattr(module, 'compute_mergers_overlap') and (self.loss_fns['overlap'] or self.report_all_metrics):
                overlap_loss += module.compute_mergers_overlap(lambda_coef_overlap=self.lambda_coef_overlap).to(similarity_loss.device)
            if hasattr(module, 'compute_weighted_overlap') and (self.loss_fns['weighted_overlap'] or self.report_all_metrics):
                weighted_overlap_loss += module.compute_weighted_overlap(lambda_coef_overlap=self.lambda_coef_overlap).to(similarity_loss.device)

        if self.loss_fns['similarity']:
            total_loss += similarity_loss 
        if self.loss_fns['l1_l2_reg']:
            total_loss += l1_l2_reg
        if self.loss_fns['overlap']:
            total_loss += overlap_loss
        if self.loss_fns['weighted_overlap']:
            total_loss += weighted_overlap_loss

        all_loss_logs['similarity_loss'] = similarity_loss
        all_loss_logs['l1_l2_reg'] = l1_l2_reg
        all_loss_logs['overlap_loss'] = overlap_loss
        all_loss_logs['weighted_overlap_loss'] = weighted_overlap_loss
        all_loss_logs['total_loss'] = total_loss

        if self.use_wandb:
            wandb.log(all_loss_logs)

        return (total_loss, merged_logits) if return_outputs else total_loss
    
    def save_model(self, output_dir=None):
        """
        Save the model and tokenizer to the specified directory.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        # Initialize a new model with the same architecture as the base model
        new_model = AutoModelForCausalLM.from_pretrained(self.base_model_path, torch_dtype=torch.bfloat16)
        
        tensors = {}
        
        # Iterate through all modules and update weights for DAMLinearLayers
        for name, module in tqdm(self.model.named_modules(), desc="Merging layers", total=len(list(self.model.named_modules()))):
            if isinstance(module, (DAMLinearLayer, DAMEmbeddingLayer, DAMRMSNorm)):
                corresponding_module = glom(new_model, name)
                # Get the merged weight and bias
                
                if isinstance(module, DAMLinearLayer):
                    merged_weight = module.get_dam_weight().to(corresponding_module.weight.device)
                if isinstance(module, DAMEmbeddingLayer):
                    merged_weight = module.get_dam_embedding().to(corresponding_module.weight.device)
                if isinstance(module, DAMRMSNorm):
                    merged_weight = module.get_dam_weight().to(corresponding_module.weight.device)
                    
                for i, merger in enumerate(module.mergers):
                    tensors[f"{name}.mergers.{i}"] = merger


                merged_bias = None
                
                if hasattr(module, 'get_dam_bias'):
                    merged_bias = module.get_dam_bias()
                    if merged_bias is not None:
                        for i, bias in enumerate(module.biases):
                            tensors[f"{name}.biases.{i}"] = bias
                        merged_bias = merged_bias.to(corresponding_module.bias.device)

                # Update the weights and bias of the corresponding layer in the new model
                corresponding_module.weight.data = merged_weight
                if merged_bias is not None:
                    corresponding_module.bias.data = merged_bias

        # Save the new model
        save_file(tensors, os.path.join(output_dir, f"mergers.safetensors"))

        new_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Merged model saved successfully at {output_dir}!")