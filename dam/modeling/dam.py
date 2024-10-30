import torch
import math
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Union
from torch.nn.parameter import Parameter
import itertools

# Base class for a layer that will merge weights from multiple models
class DAMBaseLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_models=3,
        init_merger_values=[],
        dtype=None,
        non_linearity: str = 'None',  # Option to apply non-linearity
        model_index: Optional[int] = None,
        is_embedding: bool = False,  # Flag to indicate if this is an embedding layer
        use_random_init: bool = False,  # Flag to indicate if random initialization should be used
        use_in_merging: bool = False,  # Flag to indicate if this layer should be used in merging
    ):
        super().__init__()

        # Store the number of models being merged
        self.num_models = num_models
        self.model_index = model_index

        # Store the non-linearity to be applied on the merging coefficients
        self.non_linearity = non_linearity

        # If use_random_init is True, initialize the merger values with random numbers normalized between 0 and 1
        if use_random_init:
            init_merger_values = torch.nn.functional.softmax(torch.rand(num_models), dim=0).tolist()
        # If no initial values are provided, set equal initial merger values for each model
        elif init_merger_values == []:
            init_merger_values = [1/num_models] * num_models

        # Store the initial merger values
        self.init_merger_values = init_merger_values

        # Initialize the list of merging coefficients for each model's layer
        self.mergers = nn.ParameterList([Parameter(
            torch.ones(in_features, dtype=dtype) * init_merger_values[i],
            requires_grad=use_in_merging  # Set requires_grad based on merging
        ) for i in range(num_models)])
        
        self.register_buffer('similarity_weightings', torch.zeros(math.comb(num_models, 2), in_features))

    def compute_mergers_overlap(self, lambda_coef_overlap=0.000001):
        # Check if any merger requires gradient
        if all(not merger.requires_grad for merger in self.mergers):
            device = self.mergers[0].device
            return  torch.tensor(0.0, device=device)
        
        # Initialize similarity loss
        overlap_loss = 0.0
        
        # Create all possible pairs of merging coefficients
        combinations = list(itertools.combinations([p for p in self.mergers], 2))
        num_combinations = len(combinations)
    
        if num_combinations > 0:
            overlaps = []
            for merger_a, merger_b in combinations:
                overlap = torch.sum(torch.min(torch.abs(merger_a),torch.abs(merger_b)), dim=0)

                overlaps.append(overlap)
                
            # Average the overlaps and multiply by the provided coefficient
            overlap_loss = torch.mean(torch.stack(overlaps))
            overlap_loss *= lambda_coef_overlap
    
        return overlap_loss

    # Method to compute the similarity between merging coefficients
    def compute_mergers_similarity(self, lambda_coef=None):
        # Check if any merger requires gradient
        if lambda_coef is None or all(not merger.requires_grad for merger in self.mergers):
            device = self.mergers[0].device
            return  torch.tensor(0.0, device=device)
    
        # Initialize similarity loss
        similarity_loss = 0.0
        
        # Create all possible pairs of merging coefficients
        combinations = list(itertools.combinations([p for p in self.mergers], 2))
        num_combinations = len(combinations)
    
        if num_combinations > 0:
            similarities = []
            for merger_a, merger_b in combinations:
                # Calculate cosine similarity between each pair of merging coefficients
                similarity = F.cosine_similarity(merger_a, merger_b, dim=0)
                similarities.append(similarity)
                
            # Average the similarities and multiply by the provided coefficient
            similarity_loss = torch.mean(torch.stack(similarities))
            similarity_loss *= lambda_coef
    
        return similarity_loss
    
    def compute_weighted_overlap(self, lambda_coef_overlap=1):
        merger_combinations = list(itertools.combinations([p for p in self.mergers], 2))
        merger_tensor = torch.stack([torch.stack(pair) for pair in merger_combinations])

        overlap = torch.min(torch.abs(merger_tensor), dim=1).values
        weighted_overlap = overlap * self.similarity_weightings
        
        overlaps = torch.sum(weighted_overlap, dim=1)
        
        result = torch.mean(overlaps)
        return result * lambda_coef_overlap

    # Method to compute L1 and L2 regularization on the merging coefficients
    def compute_mergers_L1_L2_reg(self, lambda_coef_l1=None, lambda_coef_l2=None):
        # Check if any merger requires gradient
        if all(not merger.requires_grad for merger in self.mergers):
            return torch.tensor(0.0, device=self.mergers[0].device)

        device = self.mergers[0].device
        l1_reg = torch.tensor(0.0, device=device)
        l2_reg = torch.tensor(0.0, device=device)

        # Calculate L1 norm for each merging coefficient in the ParameterList and sum them
        if lambda_coef_l1 is not None:
            l1_reg += sum(merger.norm(1).to(device) for merger in self.mergers) * lambda_coef_l1

        # Calculate L2 norm for each merging coefficient in the ParameterList and sum them
        if lambda_coef_l2 is not None:
            l2_reg += sum(merger.norm(2).to(device) for merger in self.mergers) * lambda_coef_l2

        # Return the combined L1 and L2 regularization loss
        return l1_reg + l2_reg

    # Method to apply the specified non-linearity to the merging coefficients
    def apply_non_linearity(self, tensor):
        if self.non_linearity == 'tanh':
            return torch.tanh(tensor)
        elif self.non_linearity == 'sigmoid':
            return torch.sigmoid(tensor)
        elif self.non_linearity == 'relu':
            return torch.relu(tensor)
        else:
            return tensor  # If non_linearity is None or unsupported, return the tensor as is

# Specialized class for an embedding layer that uses the DAMBaseLayer
class DAMEmbeddingLayer(DAMBaseLayer):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_models=3,
        init_merger_values=[],
        dtype=None,
        non_linearity: str = 'tanh',  # Option to apply non-linearity
        model_index: Optional[int] = None,
        padding_idx: Optional[int] = None,
        use_random_init: bool = False,  # Flag to indicate if random initialization should be used
        use_in_merging: bool = False,  # Flag to indicate if this layer should be used in merging
    ):
        super().__init__(
            in_features=embedding_dim,
            out_features=num_embeddings,
            num_models=num_models,
            init_merger_values=init_merger_values,
            dtype=dtype,
            non_linearity=non_linearity,
            model_index=model_index,
            is_embedding=True,  # This is an embedding layer
            use_random_init=use_random_init,  # Pass the flag to the base class
            use_in_merging=use_in_merging,  # Pass the flag to the base class
        )

        # Initialize the list of embeddings for each model's layer as nn.Embedding
        self.embeddings = nn.ParameterList([Parameter(
            torch.zeros(num_embeddings, embedding_dim, dtype=dtype),
            requires_grad=False  # Freeze embeddings
        ) for i in range(num_models)])

    # Method to compute the combined embedding for the merged layer
    def get_dam_embedding(self):
        device = self.mergers[0].device
        constrained_mergers = [self.apply_non_linearity(merger) for merger in self.mergers] if self.non_linearity else self.mergers
        # Sum the weighted contributions of each model's embedding using the (possibly constrained) merging coefficients
        return sum(merger.to(device) * embedding.to(device) for merger, embedding in zip(constrained_mergers, self.embeddings))

    # Forward pass through the DAMEmbeddingLayer
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.model_index is not None:
            # Return the output from the specified model without merging
            embedding = self.embeddings[self.model_index].to(input_ids.device)
            return F.embedding(input_ids, embedding)
        else:
            # Ensure the embeddings are on the same device as the input tensor
            embedding = self.get_dam_embedding().to(input_ids.device)
            # Perform the embedding lookup using the merged embedding
            return F.embedding(input_ids, embedding)


# Specialized class for a linear layer that uses the DAMBaseLayer
class DAMLinearLayer(DAMBaseLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_models=3,
        bias: bool=False,
        init_merger_values=[],
        dtype=None,
        non_linearity: str = 'tanh',  # Option to apply non-linearity   
        model_index: Optional[int] = None,
        use_random_init: bool = False,  # Flag to indicate if random initialization should be used
        use_in_merging: bool = True,  # Flag to indicate if this layer should be used in merging
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            num_models=num_models,
            init_merger_values=init_merger_values,
            dtype=dtype,
            non_linearity=non_linearity,
            model_index=model_index,
            is_embedding=False,  # This is not an embedding layer
            use_random_init=use_random_init,  # Pass the flag to the base class
            use_in_merging=use_in_merging,  # Pass the flag to the base class
        )

        # Initialize the list of weights for each model's layer
        self.weights = nn.ParameterList([Parameter(
            torch.zeros(out_features, in_features, dtype=dtype),
            requires_grad=False  # Freeze weights
        ) for i in range(num_models)])

        self.embedding_ties = None

        # If the layer has a bias, initialize the list of biases and bias mergers for each model
        if bias:
            self.biases = nn.ParameterList([nn.Parameter(torch.zeros(out_features, dtype=dtype), requires_grad=False) for i in range(num_models)])
            self.bias_mergers = nn.ParameterList([nn.Parameter(torch.ones(1, dtype=dtype), requires_grad=use_in_merging) for i in range(num_models)])

    # Method to compute the combined weight for the merged layer
    def get_dam_weight(self):
        if self.embedding_ties is not None:
            return self.embedding_ties.get_dam_embedding()
            
        device = self.mergers[0].device
        constrained_mergers = [self.apply_non_linearity(merger) for merger in self.mergers] if self.non_linearity else self.mergers
        # Sum the weighted contributions of each model's weight using the (possibly constrained) merging coefficients
        return sum(merger.to(device) * weight.to(device) for merger, weight in zip(constrained_mergers, self.weights))

    def tie_with_embeddings(self, embeddings: DAMEmbeddingLayer):
        self.embedding_ties = embeddings
    
    # Method to compute the combined bias for the merged layer (if bias is used)
    def get_dam_bias(self):
        if hasattr(self, 'biases'):
            device = self.bias_mergers[0].device
            constrained_bias_mergers = [self.apply_non_linearity(merger) for merger in self.bias_mergers] if self.non_linearity else self.bias_mergers
            # Sum the weighted contributions of each model's bias using the (possibly constrained) bias merging coefficients
            return sum(merger.to(device) * bias.to(device) for merger, bias in zip(constrained_bias_mergers, self.biases))
        return None

    # Forward pass through the DAMLinearLayer
    def forward(self, hidden_states: torch.Tensor) -> Union[torch.Tensor, list]:
        if self.model_index is not None:
            # Return the output from the specified model without merging
            weight = self.weights[self.model_index].to(hidden_states.device)
            bias = self.biases[self.model_index].to(hidden_states.device) if hasattr(self, 'biases') else None
            return F.linear(hidden_states, weight=weight, bias=bias)
        else:
            # Ensure the weights are on the same device as the input tensor
            weight = self.get_dam_weight().to(hidden_states.device)
            # Ensure the bias (if any) is on the same device as the input tensor
            bias = self.get_dam_bias().to(hidden_states.device) if self.get_dam_bias() is not None else None

            # Perform the linear transformation using the merged weight and bias
            return F.linear(hidden_states, weight=weight, bias=bias)

# Specialized class for a layer normalization that uses the DAMBaseLayer
class DAMRMSNorm(DAMBaseLayer):
    def __init__(
        self,
        normalized_shape: int,
        num_models=3,
        eps: float = 1e-5,
        init_merger_values=[],
        dtype=None,
        non_linearity: str = 'tanh',  # Option to apply non-linearity
        model_index: Optional[int] = None,
        use_random_init: bool = False,  # Flag to indicate if random initialization should be used
        use_in_merging: bool = False,  # Flag to indicate if this layer should be used in merging
    ):
        super().__init__(
            in_features=normalized_shape,
            out_features=normalized_shape,
            num_models=num_models,
            init_merger_values=init_merger_values,
            dtype=dtype,
            non_linearity=non_linearity,
            model_index=model_index,
            is_embedding=False,  # This is not an embedding layer
            use_random_init=use_random_init,  # Pass the flag to the base class
            use_in_merging=use_in_merging,  # Pass the flag to the base class
        )

        self.eps = eps

        # Initialize the list of weights for each model's layer normalization
        self.weights = nn.ParameterList([Parameter(
            torch.ones(normalized_shape, dtype=dtype),
            requires_grad=False  # Freeze weights
        ) for i in range(num_models)])

    # Method to compute the combined weight for the merged layer normalization
    def get_dam_weight(self):
        device = self.mergers[0].device
        constrained_mergers = [self.apply_non_linearity(merger) for merger in self.mergers] if self.non_linearity else self.mergers
        # Sum the weighted contributions of each model's weight using the (possibly constrained) merging coefficients
        return sum(merger.to(device) * weight.to(device) for merger, weight in zip(constrained_mergers, self.weights))

    # Forward pass through the DAMLayerNorm
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.model_index is not None:
            # Return the output from the specified model without merging
            weight = self.weights[self.model_index].to(hidden_states.device)
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
            return weight * hidden_states.to(input_dtype)
        else:
            # Ensure the weights are on the same device as the input tensor
            weight = self.get_dam_weight().to(hidden_states.device)
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
            return weight * hidden_states.to(input_dtype)