import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from modeling.mistral.config import MergedMistralConfig
from modeling.mistral.modeling import MergedMistralForCausalLM
from modeling.llama3.config import MergedLlamaConfig
from modeling.llama3.modeling import MergedLlamaForCausalLM
from glom import glom, Assign
from modeling.dam import DAMLinearLayer
from utils import find_linear_layers, find_embedding_layers

AutoConfig.register("mergedmistral", MergedMistralConfig)
AutoConfig.register("mergedllama", MergedLlamaConfig)
AutoModelForCausalLM.register(MergedMistralConfig, MergedMistralForCausalLM)
AutoModelForCausalLM.register(MergedLlamaConfig, MergedLlamaForCausalLM)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model, total parameters, 
    and the percentage of parameters that are trainable.
    """
    trainable_params = 0
    all_params = 0
    
    # Iterate through all parameters in the model
    for param in model.parameters():
        all_params += param.numel()  # Count total parameters
        
        if param.requires_grad:
            trainable_params += param.numel()  # Count trainable parameters
    
    # Convert to billions
    all_params_billion = all_params / 1e9
    trainable_params_billion = trainable_params / 1e9
    
    # Calculate the percentage of trainable parameters
    trainable_percentage = 100 * trainable_params / all_params
    
    # Print the results
    print(f"Total parameters: {all_params_billion:.2f} billion")
    print(f"Trainable parameters: {trainable_params_billion:.6f} billion")
    print(f"Percentage of trainable parameters: {trainable_percentage:.2f}%")


def freeze_except_mergers(model):
    # Loop through all parameters in the model
    for name, param in model.named_parameters():
        # Freeze all parameters by setting requires_grad to False
        param.requires_grad = False
        
        # Unfreeze merging weights and biases
        if "mergers" in name or "bias_mergers" in name:
            param.requires_grad = True
    
def prepare_model(model_name, cache_dir):
    print(f"Loading model from {model_name} with cache dir {cache_dir}")
    merged_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", cache_dir=cache_dir)
    print_trainable_parameters(merged_model)
    # We do not need this anymore since we freeze the params during the merging.
    #freeze_except_mergers(merged_model)
    print("####################################")
    print_trainable_parameters(merged_model)
    
    return merged_model