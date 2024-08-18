import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Union
from torch.nn.parameter import Parameter
import torch

class DAMLinearLayer(nn.Module):
    """
    DAMLayer merges linear layers from a base model and three other models using fixed task vectors 
    (differences from base weights). The trainable merging coefficients (`merger_1`, `merger_2`, `merger_3`) 
    control how these task vectors are weighted and combined during training. These coefficients allow 
    the model to learn the optimal contribution from each task vector. Additionally, similar coefficients 
    exist for biases (`bias_merger1`, `bias_merger2`, `bias_merger3`), ensuring that both weights and 
    biases are dynamically merged. Key methods include `set_forward_type` (controls weight selection), 
    `get_dam_weight` (computes merged weights), and `forward` (applies the merged transformation).
    """
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

        # keep the size of the feature
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        # The weights and biases of the base_linear layer are cloned and registered as buffers. 
        # Buffers are tensors that are not considered parameters, so they are not updated during backpropagation but are still part of the model's state
        self.register_buffer('base_weight', base_linear.weight.data.clone())
        self.register_buffer('base_bias', base_linear.bias.data.clone() if base_linear.bias is not None else None)

        # The task vectors (differences) between the weights of linear_a, linear_b, linear_c and the base_linear weights 
        # are computed and stored as buffers. These task vectors will be used to adjust the base_linear weights during the forward pass
        self.register_buffer('task_vector_1', linear_a.weight.data - self.base_weight)
        self.register_buffer('task_vector_2', linear_b.weight.data - self.base_weight)
        self.register_buffer('task_vector_3', linear_c.weight.data - self.base_weight)

        # If all the linear layers have biases, their task vectors (differences from the base bias) are computed and stored as buffers. 
        # Additionally, the merger parameters for the biases (bias_merger1, bias_merger2, bias_merger3) are initialized as learnable parameters. 
        # If biases are not present in any of the layers, None values are registered.
        if all(linear.bias is not None for linear in [base_linear, linear_a, linear_b, linear_c]):
            self.register_buffer('bias_task_vector_1', linear_a.bias.data - self.base_bias)
            self.register_buffer('bias_task_vector_2', linear_b.bias.data - self.base_bias)
            self.register_buffer('bias_task_vector_3', linear_c.bias.data - self.base_bias)
            self.bias_merger1 = nn.Parameter(torch.ones(1, device=linear_a.bias.data.device, dtype=dtype) * init_merger_value)
            self.bias_merger2 = nn.Parameter(torch.ones(1, device=linear_b.bias.data.device, dtype=dtype) * init_merger_value_2)
            self.bias_merger3 = nn.Parameter(torch.ones(1, device=linear_c.bias.data.device, dtype=dtype) * init_merger_value_3)
        else:
            self.register_buffer('bias_task_vector_1', None)
            self.register_buffer('bias_task_vector_2', None)
            self.register_buffer('bias_task_vector_3', None)
            self.register_parameter('bias_merger1', None)
            self.register_parameter('bias_merger2', None)
            self.register_parameter('bias_merger3', None)

        # The merger parameters (merger_1, merger_2, merger_3) are learnable parameters that 
        # control the weighting of the task vectors (differences from the base) from the different models. 
        # They are initialized with the provided init_merger_values.
        self.merger_1 = nn.Parameter(
            torch.ones((self.in_features,), device=linear_a.weight.data.device, dtype=dtype) * init_merger_value
        )
        self.merger_2 = nn.Parameter(
            torch.ones((self.in_features,), device=linear_b.weight.data.device, dtype=dtype) * init_merger_value_2
        )
        self.merger_3 = nn.Parameter(
            torch.ones((self.in_features,), device=linear_b.weight.data.device, dtype=dtype) * init_merger_value_3
        )
        
        #  The forward_type is initialized to "merge", which controls how the layer combines the weights during the forward pass.
        self.forward_type = "merge"

    def set_forward_type(self, type: str = "merge"):
        """
        Set the type of forward operation for the DAMLayer.

        This function controls how the weights and biases are selected during the forward pass of the DAMLayer.
        The available options allow the model to either use the base weights, one of the specific model's 
        weights, or a merged combination of all models' weights.

        Parameters:
        type (str): Specifies the type of forward pass. The options are:
            - "merge": Use the merged weights from all the models (default behavior).
            - "base": Use only the base model's weights.
            - "weight_1": Use the base model's weights combined with the task vector from `linear_a`.
            - "weight_2": Use the base model's weights combined with the task vector from `linear_b`.
            - "weight_3": Use the base model's weights combined with the task vector from `linear_c`.

        Raises:
        AssertionError: If the provided type is not one of the allowed values.
        """
        assert type in ["merge", "base", "weight_1", "weight_2", "weight_3"]
        self.forward_type = type

    def compute_mergers_similarity(self, lambda_coef_reg=None):
        """
        This method computes the cosine similarity between the merger parameters (merger_1, merger_2, merger_3).
        It returns the average similarity between these parameters, which can be used to measure how similarly the
        different models are contributing to the final merged weights.

        The goal is to make sure that the merging process doesn't overly align the contributions from each model
        but instead treats them as distinct, valuable sources of information. This helps the final model to integrate
        these diverse sources of knowledge effectively without one dominating or overlapping too much with another.

        Additionally, if a lambda_coef_reg is provided, this method applies L2 regularization to the merging coefficients
        to prevent them from becoming too large, which helps in avoiding overfitting to any particular model's task vector.

        Parameters:
        - lambda_coef_reg (float, optional): Coefficient for the L2 regularization term. If not provided, regularization is not applied.

        Returns:
        - combined_loss (float): The combined loss from cosine similarity and optional L2 regularization.
        """
        # Compute cosine similarity between the merging coefficients
        sim_12 = F.cosine_similarity(self.merger_1, self.merger_2, dim=0)
        sim_13 = F.cosine_similarity(self.merger_1, self.merger_3, dim=0)
        sim_23 = F.cosine_similarity(self.merger_2, self.merger_3, dim=0)
        
        # Average the similarities to get the overall similarity loss
        similarity_loss = (sim_12 + sim_13 + sim_23) / 3

        # Initialize regularization loss to zero
        l2_reg = 0.0

        # Apply L2 regularization if lambda_coef_reg is provided
        if lambda_coef_reg is not None:
            l2_reg = (
                self.merger_1.norm(2) +
                self.merger_2.norm(2) +
                self.merger_3.norm(2)
            ) * lambda_coef_reg

        # Combine the similarity loss and the regularization loss
        combined_loss = similarity_loss + l2_reg

        return combined_loss


    def get_dam_weight(self):
        """
        This method computes the merged weights by adding the base weights to the weighted task vectors 
        from each model (linear_a, linear_b, linear_c). The weights are combined using the merger parameters.
        """
        return self.base_weight + self.merger_1 * self.task_vector_1 + self.merger_2 * self.task_vector_2 + self.merger_3 * self.task_vector_3
    
    def get_dam_bias(self):
        if self.base_bias is not None:
            return self.base_bias + self.bias_merger1 * self.bias_task_vector_1 + self.bias_merger2 * self.bias_task_vector_2 + self.bias_merger3 * self.bias_task_vector_3
        return None

    def unfreeze(self):
        """
        Unfreezes the merger parameters so they can be trained.
        """
        self.merger_1.requires_grad = True
        self.merger_2.requires_grad = True
        self.merger_3.requires_grad = True
        if self.bias_merger1 is not None:
            self.bias_merger1.requires_grad = True
            self.bias_merger2.requires_grad = True
            self.bias_merger3.requires_grad = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the DAMLayer.

        Depending on the `forward_type` set, this method computes the output of the layer by applying
        a linear transformation to the input `hidden_states` using different combinations of weights 
        and biases.

        """
        orig_dtype = hidden_states.dtype
        dtype = self.base_weight.dtype
        
        if self.forward_type == "merge":
            weight = self.get_dam_weight()
            bias = self.get_dam_bias()
        elif self.forward_type == "base":
            weight = self.base_weight
            bias = self.base_bias
        elif self.forward_type == "weight_1":
            weight = self.base_weight + self.task_vector_1
            bias = self.base_bias + self.bias_task_vector_1 if self.base_bias is not None else None
        elif self.forward_type == "weight_2":
            weight = self.base_weight + self.task_vector_2
            bias = self.base_bias + self.bias_task_vector_2 if self.base_bias is not None else None
        elif self.forward_type == "weight_3":
            weight = self.base_weight + self.task_vector_3
            bias = self.base_bias + self.bias_task_vector_3 if self.base_bias is not None else None
        else:
            raise ValueError(self.forward_type)
        
        hidden_states = F.linear(hidden_states.to(dtype), weight=weight, bias=bias)
        return hidden_states.to(orig_dtype)


class DAMEmbeddingLayer(nn.Module):
    """
    DAMEmbeddingLayer merges embedding layers from a base model and three other models using fixed task vectors 
    (differences from base embeddings). The trainable merging coefficients (`merger_1`, `merger_2`, `merger_3`) 
    control how these task vectors are weighted and combined during training. These coefficients allow 
    the model to learn the optimal contribution from each task vector.
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
        super().__init__()

        assert (embedding_a.num_embeddings == embedding_b.num_embeddings == embedding_c.num_embeddings), "Number of embeddings must match"
        assert (embedding_a.embedding_dim == embedding_b.embedding_dim == embedding_c.embedding_dim), "Embedding dimensions must match"

        self.num_embeddings = embedding_a.num_embeddings
        self.embedding_dim = embedding_a.embedding_dim
        self.padding_idx = embedding_a.padding_idx
        self.max_norm = embedding_a.max_norm
        self.norm_type = embedding_a.norm_type
        self.scale_grad_by_freq = embedding_a.scale_grad_by_freq
        self.sparse = embedding_a.sparse

        # The weights of the embedding layers are stored as buffers.
        self.register_buffer('base_weight', embedding_a.weight.data.clone())
        self.register_buffer('task_vector_1', embedding_b.weight.data - self.base_weight)
        self.register_buffer('task_vector_2', embedding_c.weight.data - self.base_weight)

        # The merger parameters are learnable parameters that control the weighting of the task vectors.
        self.merger_1 = Parameter(
            torch.ones((self.num_embeddings,), device=embedding_a.weight.data.device, dtype=dtype) * init_merger_value
        )
        self.merger_2 = Parameter(
            torch.ones((self.num_embeddings,), device=embedding_b.weight.data.device, dtype=dtype) * init_merger_value_2
        )
        self.merger_3 = Parameter(
            torch.ones((self.num_embeddings,), device=embedding_c.weight.data.device, dtype=dtype) * init_merger_value_3
        )
        
        #  The forward_type is initialized to "merge", which controls how the layer combines the weights during the forward pass.
        self.forward_type = "merge"

    def set_forward_type(self, type: str = "merge"):
        """
        Set the type of forward operation for the DAMEmbeddingLayer.

        This function controls how the weights are selected during the forward pass of the DAMEmbeddingLayer.
        The available options allow the model to either use the base weights, one of the specific model's 
        weights, or a merged combination of all models' weights.

        Parameters:
        type (str): Specifies the type of forward pass. The options are:
            - "merge": Use the merged weights from all the models (default behavior).
            - "base": Use only the base model's weights.
            - "weight_1": Use the base model's weights combined with the task vector from `embedding_a`.
            - "weight_2": Use the base model's weights combined with the task vector from `embedding_b`.
            - "weight_3": Use the base model's weights combined with the task vector from `embedding_c`.

        Raises:
        AssertionError: If the provided type is not one of the allowed values.
        """
        assert type in ["merge", "base", "weight_1", "weight_2", "weight_3"]
        self.forward_type = type

    def get_dam_embedding_weight(self):
        """
        This method computes the merged embedding weights by adding the base weights to the weighted task vectors 
        from each model (embedding_a, embedding_b, embedding_c). The weights are combined using the merger parameters.
        """
        return self.base_weight + self.merger_1.unsqueeze(1) * self.task_vector_1 + self.merger_2.unsqueeze(1) * self.task_vector_2 + self.merger_3.unsqueeze(1) * self.task_vector_3
    
    def compute_mergers_similarity(self, lambda_coef_reg=None):
        """
        This method computes the cosine similarity between the merger parameters (merger_1, merger_2, merger_3).
        It returns the average similarity between these parameters, which can be used to measure how similarly the
        different models are contributing to the final merged weights.

        The goal is to make sure that the merging process doesn't overly align the contributions from each model
        but instead treats them as distinct, valuable sources of information. This helps the final model to integrate
        these diverse sources of knowledge effectively without one dominating or overlapping too much with another.

        Additionally, if a lambda_coef_reg is provided, this method applies L2 regularization to the merging coefficients
        to prevent them from becoming too large, which helps in avoiding overfitting to any particular model's task vector.

        Parameters:
        - lambda_coef_reg (float, optional): Coefficient for the L2 regularization term. If not provided, regularization is not applied.

        Returns:
        - combined_loss (float): The combined loss from cosine similarity and optional L2 regularization.
        """
        # Compute cosine similarity between the merging coefficients
        sim_12 = F.cosine_similarity(self.merger_1, self.merger_2, dim=0)
        sim_13 = F.cosine_similarity(self.merger_1, self.merger_3, dim=0)
        sim_23 = F.cosine_similarity(self.merger_2, self.merger_3, dim=0)
        
        # Average the similarities to get the overall similarity loss
        similarity_loss = (sim_12 + sim_13 + sim_23) / 3

        # Initialize regularization loss to zero
        l2_reg = 0.0

        # Apply L2 regularization if lambda_coef_reg is provided
        if lambda_coef_reg is not None:
            l2_reg = (
                self.merger_1.norm(2) +
                self.merger_2.norm(2) +
                self.merger_3.norm(2)
            ) * lambda_coef_reg

        # Combine the similarity loss and the regularization loss
        combined_loss = similarity_loss + l2_reg

        return combined_loss
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the DAMEmbeddingLayer.

        Depending on the `forward_type` set, this method computes the output of the layer by applying
        an embedding transformation to the input `input` using different combinations of weights.
        """
        if self.forward_type == "merge":
            weight = self.get_dam_embedding_weight()
        elif self.forward_type == "base":
            weight = self.base_weight
        elif self.forward_type == "weight_1":
            weight = self.base_weight + self.task_vector_1
        elif self.forward_type == "weight_2":
            weight = self.base_weight + self.task_vector_2
        elif self.forward_type == "weight_3":
            weight = self.base_weight + self.task_vector_3
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
