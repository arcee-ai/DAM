import torch
import torch.nn.functional as F
from transformers import Trainer

class DAMTrainer(Trainer):
    def __init__(self, model, lambda_coef=0.01, lambda_coef_reg=0.01, temperature=2.0, **kwargs):
        super().__init__(model=model, **kwargs)
        self.lambda_coef = lambda_coef
        self.lambda_coef_reg = lambda_coef_reg
        self.temperature = temperature
    
    
    def compute_loss(self, merged_model, inputs, return_outputs=False):
        # Ensure the merged_model is on the correct device
        device = merged_model.device
        
        # Dynamically identify the number of input types (e.g., input_ids_1, input_ids_2, etc.)
        input_keys = [key for key in inputs.keys() if key.startswith('input_ids_')]
        num_inputs = len(input_keys)
    
        # Prepare dictionaries to store input_ids and attention_masks
        input_ids_dict = {}
        attention_mask_dict = {}
        
        for i in range(1, num_inputs + 1):
            input_ids_dict[f'input_ids_{i}'] = inputs[f'input_ids_{i}'].to(device)
            attention_mask_dict[f'attention_mask_{i}'] = inputs[f'attention_mask_{i}'].to(device)

        # Compute logits for the merged model separately for each input
        merged_logits_dict = {}
        for i in range(1, num_inputs + 1):
            input_ids = input_ids_dict[f'input_ids_{i}']
            attention_mask = attention_mask_dict[f'attention_mask_{i}']
            merged_logits = merged_model(input_ids=input_ids, attention_mask=attention_mask).logits
            merged_logits_dict[f'merged_logits_{i}'] = merged_logits
        
        # Compute KL divergence loss between the merged model's logits and each corresponding top-K logits
        total_loss = 0.0
        for i in range(1, num_inputs + 1):
            merged_logits = merged_logits_dict[f'merged_logits_{i}']
            topk_logits = inputs[f'topk_logits_model_{i}'].to(device)
            topk_indices = inputs[f'topk_indices_model_{i}'].to(device)
            
            # Gather the logits corresponding to the top-K indices
            gathered_merged_logits = torch.gather(merged_logits, dim=-1, index=topk_indices)

            # Calculate the KL divergence loss with normalization by the input length
            kl_loss = F.kl_div(
                F.log_softmax(gathered_merged_logits / self.temperature, dim=-1),
                F.softmax(topk_logits / self.temperature, dim=-1),
                reduction='batchmean'
            ) * (self.temperature ** 2) / gathered_merged_logits.size(1)
            
            total_loss += kl_loss
        
        # Compute similarity loss and L2 regularization for merging coefficients
        similarity_loss = torch.tensor(0.0, device=device)
        l2_reg = torch.tensor(0.0, device=device)
        for module in merged_model.modules():
            if hasattr(module, 'compute_mergers_similarity'):
                similarity_loss += module.compute_mergers_similarity(self.lambda_coef).to(similarity_loss.device)
            if hasattr(module, 'compute_mergers_L2_reg'):
                l2_reg += module.compute_mergers_L2_reg(self.lambda_coef_reg).to(l2_reg.device)
    
        total_loss += similarity_loss + l2_reg

        return (total_loss, merged_logits) if return_outputs else total_loss
