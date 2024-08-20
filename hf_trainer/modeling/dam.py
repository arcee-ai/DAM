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
        in_features: int,
        out_features: int,
        init_merger_value: Optional[float] = 0.33,
        init_merger_value_2: Optional[float] = 0.33,
        init_merger_value_3: Optional[float] = 0.33,
    ):
        super().__init__()
        
        self.register_buffer('weight_1',  torch.zeros(out_features, in_features))
        self.register_buffer('weight_2', torch.zeros(out_features, in_features))
        self.register_buffer('weight_3', torch.zeros(out_features, in_features))

        # The merger parameters are learnable parameters that control the weighting of the task vectors.
        self.merger_1 = Parameter(
            torch.ones(in_features) * init_merger_value
        )
        self.merger_2 = Parameter(
            torch.ones(in_features) * init_merger_value_2
        )
        self.merger_3 = Parameter(
            torch.ones(in_features) * init_merger_value_3
        )

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
        in_features: int,
        out_features: int,
        bias: bool,
        init_merger_value: Optional[float] = 0.33,
        init_merger_value_2: Optional[float] = 0.33,
        init_merger_value_3: Optional[float] = 0.33,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            init_merger_value=init_merger_value,
            init_merger_value_2=init_merger_value_2,
            init_merger_value_3=init_merger_value_3,
        )

        if bias:
            self.register_buffer('bias_1', linear_a.bias.data)
            self.register_buffer('bias_2', linear_b.bias.data)
            self.register_buffer('bias_3', linear_c.bias.data)
            self.bias_merger1 = nn.Parameter(torch.ones(1) * init_merger_value)
            self.bias_merger2 = nn.Parameter(torch.ones(1) * init_merger_value_2)
            self.bias_merger3 = nn.Parameter(torch.ones(1) * init_merger_value_3)
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
        weight = self.get_dam_weight()
        bias = self.get_dam_bias()

        return F.linear(hidden_states, weight=weight, bias=bias)


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