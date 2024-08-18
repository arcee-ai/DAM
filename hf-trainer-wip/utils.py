import torch
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from peft.tuners.lora import QuantLinear

def find_embedding_layers(model):
    """
    Finds all embedding layers in the model.
    """
    cls = torch.nn.Embedding
    names = []
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names.append(name)
    return names

def find_all_linear_names(model):
    """
    Finds all linear layers in the model, including standard linear layers, quantized layers, 
    and custom layers like QuantLinear.
    """
    cls = (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt, torch.nn.Linear, QuantLinear)
    names = []
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names.append(name)
    return names
