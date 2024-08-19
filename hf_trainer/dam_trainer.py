import torch
import torch.nn.functional as F
from transformers import Trainer
from dam import DAMLinearLayer, DAMEmbeddingLayer
import os
from tqdm import tqdm

class DAMTrainer(Trainer):
    def compute_loss(self, model, inputs, lambda_coef=0.01, lambda_coef_reg=None, temperature=2.0, return_outputs=False):
        # Ensure all tensors are on the same device (single GPU)
        device = model.device
        
        self.lambda_coef = lambda_coef
        self.lambda_coef_reg = lambda_coef_reg
        self.temperature = temperature

        # Extract inputs for model and move them to the correct device
        input_ids_1 = inputs['input_ids_1'].to(device)
        attention_mask_1 = inputs['attention_mask_1'].to(device)
        labels_1 = inputs['labels_1'].to(device)

        input_ids_2 = inputs['input_ids_2'].to(device)
        attention_mask_2 = inputs['attention_mask_2'].to(device)
        labels_2 = inputs['labels_2'].to(device)

        input_ids_3 = inputs['input_ids_3'].to(device)
        attention_mask_3 = inputs['attention_mask_3'].to(device)
        labels_3 = inputs['labels_3'].to(device)

        # Compute logits for each input set (across batches)
        with torch.no_grad():
            model.eval()
            for module in model.modules():
                if isinstance(module, (DAMLinearLayer, DAMEmbeddingLayer)):
                    module.set_forward_type('weight_1')
            logits_1 = model(input_ids=input_ids_1, attention_mask=attention_mask_1, labels=labels_1).logits

            for module in model.modules():
                if isinstance(module, (DAMLinearLayer, DAMEmbeddingLayer)):
                    module.set_forward_type('weight_2')
            logits_2 = model(input_ids=input_ids_2, attention_mask=attention_mask_2, labels=labels_2).logits

            for module in model.modules():
                if isinstance(module, (DAMLinearLayer, DAMEmbeddingLayer)):
                    module.set_forward_type('weight_3')
            logits_3 = model(input_ids=input_ids_3, attention_mask=attention_mask_3, labels=labels_3).logits

        # Merge forward
        model.train()
        for module in model.modules():
            if isinstance(module, (DAMLinearLayer, DAMEmbeddingLayer)):
                module.set_forward_type('merge')

        merged_logits_1 = model(input_ids=input_ids_1, attention_mask=attention_mask_1, labels=labels_1).logits
        merged_logits_2 = model(input_ids=input_ids_2, attention_mask=attention_mask_2, labels=labels_2).logits
        merged_logits_3 = model(input_ids=input_ids_3, attention_mask=attention_mask_3, labels=labels_3).logits

        loss_1 = F.kl_div(
            F.log_softmax(merged_logits_1 / self.temperature, dim=-1),
            F.softmax(logits_1 / temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        loss_2 = F.kl_div(
            F.log_softmax(merged_logits_2 / self.temperature, dim=-1),
            F.softmax(logits_2 / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        loss_3 = F.kl_div(
            F.log_softmax(merged_logits_3 / self.temperature, dim=-1),
            F.softmax(logits_3 / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        similarity_loss = torch.tensor(0.0, device=device)
        similarity_reg_loss = torch.tensor(0.0, device=device)
        for module in base_model.modules():
            if isinstance(module, (DAMLinearLayer, DAMEmbeddingLayer)):
                similarity_loss += module.compute_mergers_similarity(self.lambda_coef).to(similarity_loss.device)
                similarity_reg_loss += module.compute_mergers_L2_reg(self.lambda_coef_reg).to(similarity_reg_loss.device)
         
        # Total loss (sum across the batch)
        total_loss = (loss_1 + loss_2 + loss_3) + similarity_loss + similarity_reg_loss

        print(total_loss)

        return (total_loss, merged_logits_1) if return_outputs else total_loss

    #def save_model(self, output_dir=None, _internal_call: bool = True ):
    def save_model(self, output_dir=None, **kwargs):
        # Handle any extra arguments (e.g., _internal_call) to prevent issues
        if output_dir is None:
            output_dir = self.args.output_dir

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Create a new model instance with the same architecture as the base model
        new_model = self.model.__class__.from_pretrained(self.model.config._name_or_path, torch_dtype=self.model.dtype)

        # Iterate through all modules and update weights for DAM layers
        for (name, module), (_, new_module) in tqdm(zip(self.model.named_modules(), new_model.named_modules()), 
                                                    desc="Merging layers"):
            if isinstance(module, DAMLinearLayer):
                # Get the merged weight and bias
                merged_weight = module.get_dam_weight()
                merged_bias = module.get_dam_bias()
                
                # Update the weights and bias of the corresponding layer in the new model
                new_module.weight.data = merged_weight
                if merged_bias is not None:
                    new_module.bias.data = merged_bias
                    
            if isinstance(module, DAMEmbeddingLayer):
                merged_weight = module.get_dam_embedding_weight()
                new_module.weight.data = merged_weight

        # Save the new model
        new_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        print("Merged model saved successfully!")