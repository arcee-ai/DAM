# dam.py
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, List, Union
from torch.nn.parameter import Parameter

class DAMLinearLayer(nn.Module):
    """
    Merges linear layers from a base model and multiple other models using trainable merging coefficients.
    """
    def __init__(
        self,
        base_linear: nn.Linear,
        linear_modules: List[nn.Linear],
        init_merger_values: Optional[List[float]] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.base_linear = base_linear
        self.linear_modules = linear_modules
        
        if init_merger_values is None:
            init_merger_values = [1.0 / len(linear_modules) for _ in linear_modules]
        
        self.merging_coefficients = Parameter(torch.tensor(init_merger_values, device=device, dtype=dtype))
        
        self.weight = self.get_merged_weight()
        self.bias = self.get_merged_bias() if base_linear.bias is not None else None

    def get_merged_weight(self):
        merged_weight = self.merging_coefficients[0] * self.base_linear.weight.data
        for coef, linear_module in zip(self.merging_coefficients[1:], self.linear_modules):
            merged_weight += coef * linear_module.weight.data
        return merged_weight

    def get_merged_bias(self):
        if self.base_linear.bias is not None:
            merged_bias = self.merging_coefficients[0] * self.base_linear.bias.data
            for coef, linear_module in zip(self.merging_coefficients[1:], self.linear_modules):
                merged_bias += coef * linear_module.bias.data
            return merged_bias
        return None
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight, self.bias)

class DAMEmbeddingLayer(nn.Module):
    """
    Merges embedding layers from a base model and multiple other models using trainable merging coefficients.
    """
    def __init__(
        self,
        base_embedding: nn.Embedding,
        embedding_modules: List[nn.Embedding],
        init_merger_values: Optional[List[float]] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.base_embedding = base_embedding
        self.embedding_modules = embedding_modules
        
        if init_merger_values is None:
            init_merger_values = [1.0 / len(embedding_modules) for _ in embedding_modules]
        
        self.merging_coefficients = Parameter(torch.tensor(init_merger_values, device=device, dtype=dtype))
        
        self.weight = self.get_merged_weight()

    def get_merged_weight(self):
        merged_weight = self.merging_coefficients[0] * self.base_embedding.weight.data
        for coef, embedding_module in zip(self.merging_coefficients[1:], self.embedding_modules):
            merged_weight += coef * embedding_module.weight.data
        return merged_weight
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(
            input,
            self.weight,
            self.base_embedding.padding_idx,
            self.base_embedding.max_norm,
            self.base_embedding.norm_type,
            self.base_embedding.scale_grad_by_freq,
            self.base_embedding.sparse
        )
