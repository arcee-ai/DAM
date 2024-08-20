import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Union
from torch.nn.parameter import Parameter
from transformers import AutoModelForCausalLM

class DAMBaseLayer(nn.Module):
    """
    Base class for DAM (Dynamic Alignment Merger) layers, which handles common functionality for both linear 
    and embedding layers. It merges layers from a base model and three other models using fixed weights 
    provided for each model. The trainable merging coefficients control how these weights are combined during training.
    """
    def __init__(
        self,
        weight_1: Tensor,
        weight_2: Tensor,
        weight_3: Tensor,
        init_merger_value: Optional[float] = 0.33,
        init_merger_value_2: Optional[float] = 0.33,
        init_merger_value_3: Optional[float] = 0.33,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        
        self.register_buffer('weight_1', weight_1)
        self.register_buffer('weight_2', weight_2)
        self.register_buffer('weight_3', weight_3)

        # The merger parameters are learnable parameters that control the weighting of the task vectors.
        self.merger_1 = Parameter(
            torch.ones(weight_1.size(1), device=weight_1.device, dtype=dtype) * init_merger_value
        )
        self.merger_2 = Parameter(
            torch.ones(weight_2.size(1), device=weight_2.device, dtype=dtype) * init_merger_value_2
        )
        self.merger_3 = Parameter(
            torch.ones(weight_3.size(1), device=weight_3.device, dtype=dtype) * init_merger_value_3
        )

        self.forward_type = "merge"

    def set_forward_type(self, type: str = "merge"):
        assert type in ["merge", "weight_1", "weight_2", "weight_3"]
        self.forward_type = type

    def compute_mergers_similarity(self, lambda_coef=None):
        similarity_loss = torch.tensor(0.0)

        if lambda_coef is not None:
            sim_12 = F.cosine_similarity(self.merger_1, self.merger_2, dim=0).to(similarity_loss.device)
            sim_13 = F.cosine_similarity(self.merger_1, self.merger_3, dim=0).to(similarity_loss.device)
            sim_23 = F.cosine_similarity(self.merger_2, self.merger_3, dim=0).to(similarity_loss.device)
            
            similarity_loss += (sim_12 + sim_13 + sim_23) / 3
            similarity_loss *= lambda_coef

        return similarity_loss

    def compute_mergers_L2_reg(self, lambda_coef_reg=None):
        l2_reg = torch.tensor(0.0)
        if lambda_coef_reg is not None:
            l2_reg += (
                self.merger_1.norm(2) +
                self.merger_2.norm(2) +
                self.merger_3.norm(2)
            ) * lambda_coef_reg

        return l2_reg

    def unfreeze(self):
        self.merger_1.requires_grad = True
        self.merger_2.requires_grad = True
        self.merger_3.requires_grad = True

class DAMLinearLayer(DAMBaseLayer):
    """
    DAMLinearLayer merges linear layers from a base model and three other models using fixed weights provided 
    for each model. The trainable merging coefficients control how these weights are combined during training.
    """
    def __init__(
        self,
        linear_a: nn.Linear,
        linear_b: nn.Linear,
        linear_c: nn.Linear,
        init_merger_value: Optional[float] = 0.33,
        init_merger_value_2: Optional[float] = 0.33,
        init_merger_value_3: Optional[float] = 0.33,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            weight_1=linear_a.weight.data,
            weight_2=linear_b.weight.data,
            weight_3=linear_c.weight.data,
            init_merger_value=init_merger_value,
            init_merger_value_2=init_merger_value_2,
            init_merger_value_3=init_merger_value_3,
            dtype=dtype
        )

        if all(linear.bias is not None for linear in [linear_a, linear_b, linear_c]):
            self.register_buffer('bias_1', linear_a.bias.data)
            self.register_buffer('bias_2', linear_b.bias.data)
            self.register_buffer('bias_3', linear_c.bias.data)
            self.bias_merger1 = nn.Parameter(torch.ones(1, device=linear_a.bias.data.device, dtype=dtype) * init_merger_value)
            self.bias_merger2 = nn.Parameter(torch.ones(1, device=linear_b.bias.data.device, dtype=dtype) * init_merger_value_2)
            self.bias_merger3 = nn.Parameter(torch.ones(1, device=linear_c.bias.data.device, dtype=dtype) * init_merger_value_3)
        else:
            self.register_buffer('bias_1', None)
            self.register_buffer('bias_2', None)
            self.register_buffer('bias_3', None)
            self.register_parameter('bias_merger1', None)
            self.register_parameter('bias_merger2', None)
            self.register_parameter('bias_merger3', None)

    def get_dam_weight(self):
        return self.merger_1 * self.weight_1 + self.merger_2 * self.weight_2 + self.merger_3 * self.weight_3
    
    def get_dam_bias(self):
        if self.bias_1 is not None:
            return self.bias_merger1 * self.bias_1 + self.bias_merger2 * self.bias_2 + self.bias_merger3 * self.bias_3
        return None
    

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.weight_1.dtype

        if self.forward_type == "merge":
            weight = self.get_dam_weight()
            bias = self.get_dam_bias()
        elif self.forward_type == "weight_1":
            weight = self.weight_1
            bias = self.bias_1
        elif self.forward_type == "weight_2":
            weight = self.weight_2
            bias = self.bias_2
        elif self.forward_type == "weight_3":
            weight = self.weight_3
            bias = self.bias_3
        else:
            raise ValueError(self.forward_type)

        hidden_states = F.linear(hidden_states.to(dtype).to(weight.device), weight=weight, bias=bias)
        return hidden_states.to(orig_dtype)


class DAMEmbeddingLayer(DAMBaseLayer):
    """
    DAMEmbeddingLayer merges embedding layers from a base model and three other models using fixed weights 
    provided for each model. The trainable merging coefficients control how these weights are combined during training.
    """
    def __init__(
        self,
        embedding_a: nn.Embedding,
        embedding_b: nn.Embedding,
        embedding_c: nn.Embedding,
        init_merger_value: Optional[float] = 0.33,
        init_merger_value_2: Optional[float] = 0.33,
        init_merger_value_3: Optional[float] = 0.33,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            weight_1=embedding_a.weight.data,
            weight_2=embedding_b.weight.data,
            weight_3=embedding_c.weight.data,
            init_merger_value=init_merger_value,
            init_merger_value_2=init_merger_value_2,
            init_merger_value_3=init_merger_value_3,
            dtype=dtype
        )

        self.padding_idx = embedding_a.padding_idx
        self.max_norm = embedding_a.max_norm
        self.norm_type = embedding_a.norm_type
        self.scale_grad_by_freq = embedding_a.scale_grad_by_freq
        self.sparse = embedding_a.sparse

    def get_dam_embedding_weight(self):
        return self.merger_1 * self.weight_1 + self.merger_2 * self.weight_2 + self.merger_3 * self.weight_3

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.forward_type == "merge":
            weight = self.get_dam_embedding_weight()
        elif self.forward_type == "weight_1":
            weight = self.weight_1
        elif self.forward_type == "weight_2":
            weight = self.weight_2
        elif self.forward_type == "weight_3":
            weight = self.weight_3
        else:
            raise ValueError(self.forward_type)

        return F.embedding(
            input,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse
        )

class MergedModulesMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Call the parent class initializer
        if not hasattr(self, '_merged_modules'):
            self._merged_modules = []  # Initialize merged modules list

    def register_merged_module(self, module):
        """Register a module to be managed by this class."""
        self._merged_modules.append(module)

    def merged_modules(self):
        """Return the list of merged modules."""
        return self._merged_modules

    def set_forward_type(self, forward_type):
        """Set the forward type for all registered merged modules."""
        for module in self._merged_modules:
            module.set_forward_type(forward_type)

class MergedModel(MergedModulesMixin, AutoModelForCausalLM):
    def __init__(self, *args, **kwargs):
        MergedModulesMixin.__init__(self)  # Explicitly initialize MergedModulesMixin
        AutoModelForCausalLM.__init__(self, *args, **kwargs)  # Initialize the base model
