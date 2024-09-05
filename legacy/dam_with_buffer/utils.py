import torch

def find_linear_layers(model):
    return [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]

def find_embedding_layers(model):
    return [name for name, module in model.named_modules() if isinstance(module, torch.nn.Embedding)]
