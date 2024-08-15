import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Union

class DAMLayer(nn.Module):
    def __init__(
        self,
        base_linear: nn.Linear,
        linear_a: nn.Linear,
        linear_b: nn.Linear,
        linear_c: nn.Linear,
        init_merger_value: Optional[float] = 0.33,
        init_merger_value_2: Optional[float] = 0.33,
        init_merger_value_3: Optional[float] = 0.33,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        assert (base_linear.in_features == linear_a.in_features == linear_b.in_features == linear_c.in_features), "Input features must match"
        assert (base_linear.out_features == linear_a.out_features == linear_b.out_features == linear_c.out_features), "Output features must match"

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        self.register_buffer('base_weight', base_linear.weight.data.clone())
        self.register_buffer('base_bias', base_linear.bias.data.clone() if base_linear.bias is not None else None)

        self.register_buffer('residual_1', linear_a.weight.data - self.base_weight)
        self.register_buffer('residual_2', linear_b.weight.data - self.base_weight)
        self.register_buffer('residual_3', linear_c.weight.data - self.base_weight)

        if all(linear.bias is not None for linear in [base_linear, linear_a, linear_b, linear_c]):
            self.register_buffer('bias_residual_1', linear_a.bias.data - self.base_bias)
            self.register_buffer('bias_residual_2', linear_b.bias.data - self.base_bias)
            self.register_buffer('bias_residual_3', linear_c.bias.data - self.base_bias)
            self.bias_merger1 = nn.Parameter(torch.ones(1, device=linear_a.bias.data.device, dtype=dtype) * init_merger_value)
            self.bias_merger2 = nn.Parameter(torch.ones(1, device=linear_b.bias.data.device, dtype=dtype) * init_merger_value_2)
            self.bias_merger3 = nn.Parameter(torch.ones(1, device=linear_c.bias.data.device, dtype=dtype) * init_merger_value_3)
        else:
            self.register_buffer('bias_residual_1', None)
            self.register_buffer('bias_residual_2', None)
            self.register_buffer('bias_residual_3', None)
            self.register_parameter('bias_merger1', None)
            self.register_parameter('bias_merger2', None)
            self.register_parameter('bias_merger3', None)

        self.merger_1 = nn.Parameter(
            torch.ones((self.in_features,), device=linear_a.weight.data.device, dtype=dtype) * init_merger_value
        )
        self.merger_2 = nn.Parameter(
            torch.ones((self.in_features,), device=linear_b.weight.data.device, dtype=dtype) * init_merger_value_2
        )
        self.merger_3 = nn.Parameter(
            torch.ones((self.in_features,), device=linear_b.weight.data.device, dtype=dtype) * init_merger_value_3
        )
        
        self.forward_type = "merge"

    def set_forward_type(self, type: str = "merge"):
        assert type in ["merge", "base", "weight_1", "weight_2", "weight_3"]
        self.forward_type = type

    def compute_mergers_similarity(self):
        sim_12 = F.cosine_similarity(self.merger_1, self.merger_2, dim=0)
        sim_13 = F.cosine_similarity(self.merger_1, self.merger_3, dim=0)
        sim_23 = F.cosine_similarity(self.merger_2, self.merger_3, dim=0)
    
        return (sim_12 + sim_13 + sim_23) / 3

    def get_dam_weight(self):
        return self.base_weight + self.merger_1 * self.residual_1 + self.merger_2 * self.residual_2 + self.merger_3 * self.residual_3
    
    def get_dam_bias(self):
        if self.base_bias is not None:
            return self.base_bias + self.bias_merger1 * self.bias_residual_1 + self.bias_merger2 * self.bias_residual_2 + self.bias_merger3 * self.bias_residual_3
        return None

    def unfreeze(self):
        self.merger_1.requires_grad = True
        self.merger_2.requires_grad = True
        self.merger_3.requires_grad = True
        if self.bias_merger1 is not None:
            self.bias_merger1.requires_grad = True
            self.bias_merger2.requires_grad = True
            self.bias_merger3.requires_grad = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.base_weight.dtype
        if self.forward_type == "merge":
            weight = self.get_dam_weight()
            bias = self.get_dam_bias()
        elif self.forward_type == "base":
            weight = self.base_weight
            bias = self.base_bias
        elif self.forward_type == "weight_1":
            weight = self.base_weight + self.residual_1
            bias = self.base_bias + self.bias_residual_1 if self.base_bias is not None else None
        elif self.forward_type == "weight_2":
            weight = self.base_weight + self.residual_2
            bias = self.base_bias + self.bias_residual_2 if self.base_bias is not None else None
        elif self.forward_type == "weight_3":
            weight = self.base_weight + self.residual_3
            bias = self.base_bias + self.bias_residual_3 if self.base_bias is not None else None
        else:
            raise ValueError(self.forward_type)
        
        hidden_states = F.linear(hidden_states.to(dtype), weight=weight, bias=bias)
        return hidden_states.to(orig_dtype)
