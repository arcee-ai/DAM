import torch
import torch.nn.functional as F
from transformers import Trainer, AutoModelForCausalLM
from modeling.dam import DAMLinearLayer
from tqdm import tqdm
try:
    import wandb
except:
    pass

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

def entropy_loss(logits, attention_mask, non_padded_tokens, temperature=1.0):
    # Apply softmax to the logits
    probabilities = F.softmax(logits / temperature, dim=-1)
    
    # Compute the entropy
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=-1)
    
    # Mask out the padding tokens in the entropy
    masked_entropy = entropy * attention_mask
    
    # Compute the total entropy loss, considering only non-padded tokens
    total_entropy_loss = masked_entropy.sum() * (temperature ** 2)
    
    # Normalize by the number of non-padded tokens
    return total_entropy_loss / non_padded_tokens

def mse_loss(logits, target_logits, non_padded_tokens):
    # Compute the MSE loss
    mse_loss = F.mse_loss(logits, target_logits, reduction='sum')
    
    # Normalize by the number of non-padded tokens
    return mse_loss / non_padded_tokens

def compute_individual_logit_losses(merged_logits, 
                                    topk_logits, 
                                    topk_indices, 
                                    attention_mask, 
                                    non_padded_tokens, 
                                    temperature, 
                                    use_kl, 
                                    use_mse, 
                                    use_entropy, 
                                    loss_logs, 
                                    model_index=None):
    masked_merged_logits = merged_logits * attention_mask.unsqueeze(-1)
    masked_topk_target_logits = topk_logits * attention_mask.unsqueeze(-1)
    gathered_masked_merged_logits = torch.gather(masked_merged_logits, dim=-1, index=topk_indices)

    total_loss = 0.0

    if use_kl:
        kl_loss = kl_divergence_loss(gathered_masked_merged_logits, 
                                     masked_topk_target_logits, 
                                     non_padded_tokens, 
                                     temperature=temperature)
        loss_logs[f'kl_loss' + (f'_model_{model_index}' if model_index is not None else '')] = kl_loss
        total_loss += kl_loss

    if use_mse:
        mse_loss_value = mse_loss(gathered_masked_merged_logits,
                                  masked_topk_target_logits, 
                                  non_padded_tokens)
        loss_logs[f'mse_loss' + (f'_model_{model_index}' if model_index is not None else '')] = mse_loss_value
        total_loss += mse_loss_value

    if use_entropy:
        e_loss = entropy_loss(masked_merged_logits, 
                              attention_mask, 
                              non_padded_tokens,
                              temperature=temperature)
        loss_logs[f'entropy_loss' + (f'_model_{model_index}' if model_index is not None else '')] = e_loss
        total_loss += e_loss

    return total_loss

class DAMTrainer(Trainer):
    def __init__(self, model, lambda_coef=0.01, 
                 lambda_coef_l1=None,  # Added L1 regularization coefficient
                 lambda_coef_l2=0.01,  # Added L2 regularization coefficient
                 temperature=2.0, 
                 use_kl=True, 
                 use_mse=False, 
                 use_entropy=False,
                 base_model_path=None, 
                 use_wandb=False,
                 generate_logits_on_fly=False,  # New parameter to control logits generation
                 **kwargs):
        super().__init__(model=model, **kwargs)
        self.lambda_coef = lambda_coef
        self.lambda_coef_l1 = lambda_coef_l1  # Initialize L1 regularization coefficient
        self.lambda_coef_l2 = lambda_coef_l2  # Initialize L2 regularization coefficient
        self.temperature = temperature

        self.use_kl = use_kl
        self.use_mse = use_mse
        self.use_entropy = use_entropy

        self.base_model_path = base_model_path
        self.use_wandb = use_wandb
        self.generate_logits_on_fly = generate_logits_on_fly  # Initialize the new parameter

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
        
        total_loss = 0.0
        loss_logs = {}

        # Compute KL divergence loss between the merged model's logits and each corresponding top-K logits
        for i in range(1, num_inputs + 1):
            merged_logits = merged_logits_dict[f'merged_logits_{i}']
            attention_mask = attention_mask_dict[f'attention_mask_{i}']
            non_padded_tokens = attention_mask.sum().item()

            if self.generate_logits_on_fly:
                for model_index in range(merged_model.num_models):
                    model_logits = merged_model(input_ids=input_ids_dict[f'input_ids_{i}'], attention_mask=attention_mask_dict[f'attention_mask_{i}'], model_index=model_index).logits
                    topk_logits = torch.topk(model_logits, k=model_logits.size(-1), dim=-1).values
                    topk_indices = torch.topk(model_logits, k=model_logits.size(-1), dim=-1).indices
                    total_loss += compute_individual_logit_losses(merged_logits, topk_logits, topk_indices, attention_mask, non_padded_tokens, self.temperature, self.use_kl, self.use_mse, self.use_entropy, loss_logs, model_index)
            else:
                topk_logits = inputs[f'topk_logits_model_{i}'].to(device)
                topk_indices = inputs[f'topk_indices_model_{i}'].to(device)
                total_loss += compute_individual_logit_losses(merged_logits, topk_logits, topk_indices, attention_mask, non_padded_tokens, self.temperature, self.use_kl, self.use_mse, self.use_entropy, loss_logs)

        # Compute similarity loss and L2 regularization for merging coefficients
        similarity_loss = torch.tensor(0.0, device=device)
        l1_l2_reg = torch.tensor(0.0, device=device)
        for module in merged_model.modules():
            if hasattr(module, 'compute_mergers_similarity'):
                similarity_loss += module.compute_mergers_similarity(self.lambda_coef).to(similarity_loss.device)
            if hasattr(module, 'compute_mergers_L2_reg'):
                l1_l2_reg += module.compute_mergers_L1_L2_reg(
                    lambda_coef_l1=self.lambda_coef_l1, 
                    lambda_coef_l2=self.lambda_coef_l2
                ).to(l1_l2_reg.device)

        loss_logs['similarity_loss'] = similarity_loss
        loss_logs['l1_l2_reg'] = l1_l2_reg

        total_loss += similarity_loss + l1_l2_reg

        if self.use_wandb:
            wandb.log(loss_logs)

        return (total_loss, merged_logits) if return_outputs else total_loss
    
    def save_model(self, output_dir=None):
        """
        Save the model and tokenizer to the specified directory.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        # Initialize a new model with the same architecture as the base model
        new_model = AutoModelForCausalLM.from_pretrained(self.base_model_path, torch_dtype=torch.bfloat16)

        # Iterate through all modules and update weights for DAMLinearLayers
        for (name, module), (_, new_module) in tqdm(zip(self.model.named_modules(), new_model.named_modules()), 
                                                    desc="Merging layers"):
            if isinstance(module, DAMLinearLayer) and isinstance(new_module, torch.nn.Linear):
                # Get the merged weight and bias
                merged_weight = module.get_dam_weight().to(new_module.weight.device)
                merged_bias = module.get_dam_bias()
                if merged_bias is not None:
                    merged_bias = merged_bias.to(new_module.bias.device)

                # Update the weights and bias of the corresponding layer in the new model
                new_module.weight.data = merged_weight
                if merged_bias is not None:
                    new_module.bias.data = merged_bias

        # Save the new model
        new_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Merged model saved successfully at {output_dir}!")