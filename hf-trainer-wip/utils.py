import torch
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_linear_embedding_layers(model_type):
    if model_type == "gpt_neox":
        return ["embed_in", "embed_out"]
    if model_type == "falcon":
        return ["word_embeddings", "lm_head"]
    return ["embed_tokens", "lm_head"]

def find_all_linear_names(model):
    cls = (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt, torch.nn.Linear)
    names = []
    for name, module in model.named_modules():
        if isinstance(module, cls) or "Linear" in module.__class__.__name__:
            names.append(name)
    return names
