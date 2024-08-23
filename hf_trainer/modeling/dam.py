import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Union
from torch.nn.parameter import Parameter
import itertools

class DAMBaseLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_models=3,
        init_merger_values=[],
        dtype=None,
    ):
        super().__init__()

        self.num_models = num_models

        if init_merger_values == []:
            init_merger_values = [1/num_models] * num_models

        self.weights = nn.ParameterList([Parameter(
            torch.zeros(out_features, in_features, dtype=dtype) * init_merger_values[i]
        ) for i in range(num_models)])

        self.mergers = nn.ParameterList([Parameter(
            torch.ones(in_features, dtype=dtype) * init_merger_values[i]
        ) for i in range(num_models)])

    def compute_mergers_similarity(self, lambda_coef=None):
        if lambda_coef is None:
            return 0.0
    
        similarity_loss = 0.0
        combinations = list(itertools.combinations([p for p in self.mergers], 2))
        num_combinations = len(combinations)
    
        if num_combinations > 0:
            similarities = []
            for merger_a, merger_b in combinations:
                similarity = F.cosine_similarity(merger_a, merger_b, dim=0)
                similarities.append(similarity)
                
            similarity_loss = torch.mean(torch.stack(similarities))
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
        for merger in self.mergers:
            merger.requires_grad = True

class DAMLinearLayer(DAMBaseLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_models=3,
        bias: bool=False,
        init_merger_values=[],
        dtype=None
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            num_models=num_models,
            init_merger_values=init_merger_values,
            dtype=dtype,
        )

        if init_merger_values == []:
            init_merger_values = [1/num_models] * num_models

        if bias:
            self.biases = nn.ParameterList([nn.Parameter(torch.zeros(out_features, dtype=dtype) * init_merger_values[i]) for i in range(num_models)])
            self.bias_mergers = nn.ParameterList([nn.Parameter(torch.ones(1, dtype=dtype) * init_merger_values[i]) for i in range(num_models)])

    def get_dam_weight(self):
        device = self.mergers[0].device
        return sum(merger.to(device) * weight.to(device) for merger, weight in zip(self.weights, self.mergers))
    
    def get_dam_bias(self):
        if hasattr(self, 'bias_0'):
            device = self.bias_mergers[0].device
            return sum(merger.to(device) * bias.to(device) for merger, bias in zip(self.bias_mergers, self.biases))
        return None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Ensure hidden_states are on the same device as the weights
        weight = self.get_dam_weight().to(hidden_states.device)
        bias = self.get_dam_bias().to(hidden_states.device) if self.get_dam_bias() is not None else None

        return F.linear(hidden_states, weight=weight, bias=bias)
