import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from modeling.modeling import MergedMistralForCausalLM
from glom import glom, Assign
from modeling.dam import DAMLinearLayer
from utils import find_linear_layers, find_embedding_layers

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for module in model.modules():
        for param in module.parameters():        
            all_param += param.numel()
            
            if param.requires_grad:
                trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    
def prepare_model(MODEL_ID, apply_to_embeddings=False):
    #merged_model = MergedMistralForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    # we can't do this auto when we are using deepspeed.
    merged_model = MergedMistralForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)

    # Freeze all parameters except for the specific layers
    for param in merged_model.parameters():
        param.requires_grad = False
    
    for module in merged_model.modules():
        if isinstance(module, DAMLinearLayer):
            for param in module.parameters():
                param.requires_grad = True
    
    print_trainable_parameters(merged_model)

    return merged_model
