from transformers import Trainer
import torch.nn.functional as F
import torch

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(CustomTrainer, self).__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")
        
        # Split inputs for each model
        input_a, input_b, input_c = input_ids.chunk(3)
        labels_a, labels_b, labels_c = labels.chunk(3)
        
        # Generate original logits for each prompt
        original_logits_a = self._get_logits(model, input_a, labels_a, 'weight_1')
        original_logits_b = self._get_logits(model, input_b, labels_b, 'weight_2')
        original_logits_c = self._get_logits(model, input_c, labels_c, 'weight_3')
        
        # Generate merged logits
        merged_logits_a = self._get_logits(model, input_a, labels_a, 'merge')
        merged_logits_b = self._get_logits(model, input_b, labels_b, 'merge')
        merged_logits_c = self._get_logits(model, input_c, labels_c, 'merge')
        
        # Calculate KL divergence loss
        loss_a = self._kl_div_loss(merged_logits_a, original_logits_a, labels_a)
        loss_b = self._kl_div_loss(merged_logits_b, original_logits_b, labels_b)
        loss_c = self._kl_div_loss(merged_logits_c, original_logits_c, labels_c)

        # Custom similarity loss for DAMLayers
        similarity_loss = self._compute_similarity_loss(model)
        
        total_loss = loss_a + loss_b + loss_c + similarity_loss
        
        return (total_loss, merged_logits_a) if return_outputs else total_loss

    def _get_logits(self, model, input_ids, labels, forward_type):
        for module in model.modules():
            if isinstance(module, DAMLayer):
                module.set_forward_type(forward_type)
        return model(input_ids=input_ids, labels=labels).logits
    
    def _kl_div_loss(self, merged_logits, original_logits, labels, temperature=2.0):
        loss = F.kl_div(
            F.log_softmax(merged_logits / temperature, dim=-1),
            F.softmax(original_logits / temperature, dim=-1),
            reduction='batchmean'
        ) * (temperature ** 2) / len(labels)
        return loss

    def _compute_similarity_loss(self, model):
        similarity_loss = torch.tensor(0.0, device=model.device)
        for module in model.modules():
            if isinstance(module, DAMLayer):
                similarity_loss += module.compute_mergers_similarity()
        return similarity_loss
