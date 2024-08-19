import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from glom import glom, Assign
from dam import DAMLinearLayer, DAMEmbeddingLayer
from utils import find_linear_layers, find_embedding_layers

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

    # Truncate model_C's layers to match the base model's vocabulary size
    truncated_lm_head = torch.nn.Linear(in_features=4096, out_features=32000, bias=False)
    truncated_embed = torch.nn.Embedding(num_embeddings=32000, embedding_dim=4096)

    with torch.no_grad():
        truncated_lm_head.weight.data = model_C.lm_head.weight.data[:32000, :]
        truncated_embed.weight.data = model_C.model.embed_tokens.weight.data[:32000, :]

    assign_embed = Assign('model.embed_tokens', truncated_embed)
    assign_lm_head = Assign('lm_head', truncated_lm_head)
    glom(model_C, assign_embed)
    glom(model_C, assign_lm_head)

    # Prepare DAM layers
    linear_modules = find_linear_layers(base_model)
    embedding_modules = find_embedding_layers(base_model) if apply_to_embeddings else []

    for m in linear_modules + embedding_modules:
        module_a = glom(model_A, m)
        module_b = glom(model_B, m)
        module_c = glom(model_C, m)

        if isinstance(module_a, torch.nn.Embedding):
            dam_layer = DAMEmbeddingLayer(
                embedding_a=module_a,
                embedding_b=module_b,
                embedding_c=module_c,
                device=base_model.device,
                dtype=module_a.weight.dtype
            )
        else:
            dam_layer = DAMLinearLayer(
                base_linear=glom(base_model, m),
                linear_a=module_a,
                linear_b=module_b,
                linear_c=module_c,
                device=base_model.device,
                dtype=module_a.weight.dtype
            )

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

    return base_model, tokenizer
