import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from glom import glom, Assign
from dam import DAMLinearLayer, DAMEmbeddingLayer
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
    
def prepare_model(MODEL_ID, MODEL_ID_A, MODEL_ID_B, MODEL_ID_C, apply_to_embeddings=True):
    # Load models
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model_A = AutoModelForCausalLM.from_pretrained(MODEL_ID_A, torch_dtype=torch.bfloat16, device_map="auto")
    model_B = AutoModelForCausalLM.from_pretrained(MODEL_ID_B, torch_dtype=torch.bfloat16, device_map="auto")
    model_C = AutoModelForCausalLM.from_pretrained(MODEL_ID_C, torch_dtype=torch.bfloat16, device_map="auto")
    # base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    # model_A = AutoModelForCausalLM.from_pretrained(MODEL_ID_A, torch_dtype=torch.bfloat16)
    # model_B = AutoModelForCausalLM.from_pretrained(MODEL_ID_B, torch_dtype=torch.bfloat16)
    # model_C = AutoModelForCausalLM.from_pretrained(MODEL_ID_C, torch_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    # Prepare DAM layers
    linear_modules = find_linear_layers(base_model)
    embedding_modules = find_embedding_layers(base_model) if apply_to_embeddings else []
    
    for m in (linear_modules + embedding_modules):
        module_a = glom(model_A, m)
        module_b = glom(model_B, m)
        module_c = glom(model_C, m)
    
        dam_layer = None
        
        if isinstance(module_a, torch.nn.Embedding):
            dam_layer = DAMEmbeddingLayer(
                embedding_a=module_a,
                embedding_b=module_b,
                embedding_c=module_c,
                device=module_c.weight.device,
                dtype=module_a.weight.dtype
            )
        else:
            dam_layer = DAMLinearLayer(
                linear_a=module_a,
                linear_b=module_b,
                linear_c=module_c,
                device=module_a.weight.device,
                dtype=module_a.weight.dtype
            )
    
        if dam_layer is not None:
            assign = Assign(m, dam_layer)
            glom(base_model, assign)

    # Freeze all parameters except for the specific layers
    for module in base_model.modules():
        if isinstance(module, (DAMLinearLayer, DAMEmbeddingLayer)):
            for param in module.parameters():
                param.requires_grad = True
        else:
            for param in module.parameters():
                param.requires_grad = False

    print_trainable_parameters(base_model)


    # We no longer need the weight source models, this will free up VRAM for training

    model_A.cpu()
    model_B.cpu()
    model_C.cpu()
    
    del model_A, model_B, model_C
    gc.collect()
    torch.cuda.empty_cache()


    return base_model, tokenizer
