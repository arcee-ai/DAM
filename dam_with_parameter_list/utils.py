import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralRMSNorm

def find_linear_layers(model):
    return [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]

def find_embedding_layers(model):
    return [name for name, module in model.named_modules() if isinstance(module, torch.nn.Embedding)]

def find_norm_layers(model):
    return [name for name, module in model.named_modules() if isinstance(module, LlamaRMSNorm) or isinstance(module, MistralRMSNorm)]