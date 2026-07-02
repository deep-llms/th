"""Inject / save / load / remove EmbHub on a HuggingFace causal LM.

Supports hub types:
- "v3": gate + add with decoupled keys/values (V3/V4/V5)
- "v2_concat": concat + linear (V2/V2b)
- "v2_topk": top-k anchor concat (V2c/V2c+tail/V2c+buckets)
- "v6": stochastic replacement (V6-mix)
- "v6f": V6 + scarcity + norm-capped residual (V6f factorized)

Supports placement at:
- "embedding": after the token embedding layer
- "mid": after a specific transformer decoder layer

Usage:
    from model_wrapper_v3 import inject_embhub_v3, save_embhub_v3, load_model_with_embhub_v3

    # V3 at embedding layer
    hub = inject_embhub_v3(model, hub_type="v3", placement="embedding")

    # V5 at mid-layer 10
    hub = inject_embhub_v3(model, hub_type="v3", placement="mid", layer_idx=10)

    # V2 concat at embedding
    hub = inject_embhub_v3(model, hub_type="v2_concat", placement="embedding")

    # V2b concat+MLP
    hub = inject_embhub_v3(model, hub_type="v2_concat", use_mlp=True)
"""

import json
import os

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig

from hub_layer_v3 import EmbHubV3
from hub_layer_v2_concat import EmbHubV2Concat
from hub_layer_v2_topk import EmbHubV2TopK
from hub_layer_v6 import EmbHubV6

EMBHUB_V3_WEIGHTS_NAME = "embhub_v3.pt"
EMBHUB_V3_CONFIG_NAME = "embhub_v3_config.json"


def _get_transformer_layers(model: nn.Module):
    """Get the list of transformer decoder layers from a HF model."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(
        "Cannot find transformer layers. Supported: model.model.layers (Qwen/Llama/Mistral) "
        "or model.transformer.h (GPT-2/GPT-Neo)"
    )


def inject_embhub_v3(
    model: nn.Module,
    hub_type: str = "v3",
    num_embeddings: int = 1000,
    num_heads: int = 1,
    gate_bias_init: float = -5.0,
    use_mlp: bool = False,
    top_k: int = 10,
    weighting: str = "raw_softmax",
    tail_mode: str = "none",
    num_buckets: int = 10,
    r_budget: float = 0.3,
    p_only: float = 0.10,
    p_both: float = 0.40,
    anneal_steps: int = 2000,
    placement: str = "embedding",
    layer_idx: int = 10,
    freeze_base: bool = False,
):
    if hasattr(model, "embhub"):
        raise ValueError("Model already has an EmbHub layer injected")

    if placement == "embedding":
        target = model.get_input_embeddings()
        embedding_dim = target.embedding_dim
        reference_weight = target.weight.data
    elif placement == "mid":
        layers = _get_transformer_layers(model)
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"layer_idx={layer_idx} out of range [0, {len(layers)})")
        target = layers[layer_idx]
        config = model.config
        embedding_dim = config.hidden_size
        reference_weight = None
    else:
        raise ValueError(f"placement must be 'embedding' or 'mid', got '{placement}'")

    if hub_type == "v3":
        hub = EmbHubV3(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            num_heads=num_heads,
            gate_bias_init=gate_bias_init,
            reference_weight=reference_weight,
        )
    elif hub_type == "v2_concat":
        hub = EmbHubV2Concat(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            use_mlp=use_mlp,
            reference_weight=reference_weight,
        )
    elif hub_type == "v2_topk":
        hub = EmbHubV2TopK(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            top_k=top_k,
            weighting=weighting,
            tail_mode=tail_mode,
            num_buckets=num_buckets,
            reference_weight=reference_weight,
        )
    elif hub_type in ("v6", "v6f"):
        hub = EmbHubV6(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            top_k=top_k,
            use_residual_cap=(hub_type == "v6f"),
            r_budget=r_budget,
            p_only=p_only,
            p_both=p_both,
            anneal_steps=anneal_steps,
            reference_weight=reference_weight,
        )
    else:
        raise ValueError(f"hub_type must be 'v3', 'v2_concat', 'v2_topk', 'v6', or 'v6f', got '{hub_type}'")
    hub.to(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)

    model.embhub = hub
    model._embhub_hub_type = hub_type
    model._embhub_placement = placement
    model._embhub_layer_idx = layer_idx

    if placement == "embedding":
        handle = target.register_forward_hook(lambda mod, inp, out: hub(out))
    else:
        def mid_layer_hook(mod, inp, out):
            if isinstance(out, tuple):
                modified = hub(out[0])
                return (modified,) + out[1:]
            return hub(out)
        handle = target.register_forward_hook(mid_layer_hook)

    model._embhub_hook_handle = handle

    if freeze_base:
        for name, param in model.named_parameters():
            if "embhub" not in name:
                param.requires_grad = False

    return hub


def remove_embhub_v3(model: nn.Module) -> None:
    if hasattr(model, "_embhub_hook_handle"):
        model._embhub_hook_handle.remove()
        del model._embhub_hook_handle
    if hasattr(model, "embhub"):
        del model.embhub
    if hasattr(model, "_embhub_hub_type"):
        del model._embhub_hub_type
    if hasattr(model, "_embhub_placement"):
        del model._embhub_placement
    if hasattr(model, "_embhub_layer_idx"):
        del model._embhub_layer_idx


def save_embhub_v3(model: nn.Module, save_directory: str) -> None:
    if not hasattr(model, "embhub"):
        raise ValueError("Model does not have an EmbHub V3 layer to save")

    os.makedirs(save_directory, exist_ok=True)

    hub = model.embhub
    state_dict = {k: v.float() for k, v in hub.state_dict().items()}
    torch.save(state_dict, os.path.join(save_directory, EMBHUB_V3_WEIGHTS_NAME))

    hub_type = getattr(model, "_embhub_hub_type", "v3")
    config = {
        "hub_type": hub_type,
        "num_embeddings": hub.num_embeddings,
        "embedding_dim": hub.embedding_dim,
        "placement": getattr(model, "_embhub_placement", "embedding"),
        "layer_idx": getattr(model, "_embhub_layer_idx", 0),
    }
    if hub_type == "v3":
        config["num_heads"] = hub.num_heads
    elif hub_type == "v2_concat":
        config["use_mlp"] = hub.use_mlp
    elif hub_type == "v2_topk":
        config["top_k"] = hub.top_k
        config["weighting"] = hub.weighting
        config["tail_mode"] = hub.tail_mode
        config["num_buckets"] = hub.num_buckets
    elif hub_type in ("v6", "v6f"):
        config["top_k"] = hub.top_k
        config["use_residual_cap"] = hub.use_residual_cap
        config["r_budget"] = hub.r_budget
        config["p_only"] = hub.p_only
        config["p_both"] = hub.p_both
        config["anneal_steps"] = hub.anneal_steps
    with open(os.path.join(save_directory, EMBHUB_V3_CONFIG_NAME), "w") as f:
        json.dump(config, f, indent=2)


def load_model_with_embhub_v3(
    model_name_or_path: str,
    hub_type: str = "v3",
    num_embeddings: int = 1000,
    num_heads: int = 1,
    gate_bias_init: float = -5.0,
    use_mlp: bool = False,
    top_k: int = 10,
    weighting: str = "raw_softmax",
    tail_mode: str = "none",
    num_buckets: int = 10,
    r_budget: float = 0.3,
    p_only: float = 0.10,
    p_both: float = 0.40,
    anneal_steps: int = 2000,
    placement: str = "embedding",
    layer_idx: int = 10,
    freeze_base: bool = False,
    **model_kwargs,
) -> tuple:
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

    config_path = os.path.join(model_name_or_path, EMBHUB_V3_CONFIG_NAME)
    weights_path = os.path.join(model_name_or_path, EMBHUB_V3_WEIGHTS_NAME)

    if os.path.isfile(config_path) and os.path.isfile(weights_path):
        with open(config_path) as f:
            hub_config = json.load(f)
        hub = inject_embhub_v3(
            model,
            hub_type=hub_config.get("hub_type", "v3"),
            num_embeddings=hub_config["num_embeddings"],
            num_heads=hub_config.get("num_heads", 1),
            gate_bias_init=gate_bias_init,
            use_mlp=hub_config.get("use_mlp", False),
            top_k=hub_config.get("top_k", 10),
            weighting=hub_config.get("weighting", "raw_softmax"),
            tail_mode=hub_config.get("tail_mode", "none"),
            num_buckets=hub_config.get("num_buckets", 10),
            r_budget=hub_config.get("r_budget", 0.3),
            p_only=hub_config.get("p_only", 0.10),
            p_both=hub_config.get("p_both", 0.40),
            anneal_steps=hub_config.get("anneal_steps", 2000),
            placement=hub_config.get("placement", "embedding"),
            layer_idx=hub_config.get("layer_idx", 10),
            freeze_base=freeze_base,
        )
        hub.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    else:
        hub = inject_embhub_v3(
            model,
            hub_type=hub_type,
            num_embeddings=num_embeddings,
            num_heads=num_heads,
            gate_bias_init=gate_bias_init,
            use_mlp=use_mlp,
            top_k=top_k,
            weighting=weighting,
            tail_mode=tail_mode,
            num_buckets=num_buckets,
            r_budget=r_budget,
            p_only=p_only,
            p_both=p_both,
            anneal_steps=anneal_steps,
            placement=placement,
            layer_idx=layer_idx,
            freeze_base=freeze_base,
        )

    return model, hub
