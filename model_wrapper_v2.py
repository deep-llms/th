import json
import os

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig

from hub_layer_v2 import EmbHub

EMBHUB_WEIGHTS_NAME = "embhub.pt"
EMBHUB_CONFIG_NAME = "embhub_config.json"


def inject_embhub(
    model: nn.Module,
    num_embeddings: int = 1000,
    alpha: float = 0.05,
    freeze_base: bool = False,
) -> EmbHub:
    """Inject an EmbHub layer after the embedding layer of any HuggingFace causal LM.

    Returns the EmbHub module so callers can access it directly.
    """
    if hasattr(model, "embhub"):
        raise ValueError("Model already has an EmbHub layer injected")

    embedding = model.get_input_embeddings()

    hub = EmbHub(
        embedding_dim=embedding.embedding_dim,
        num_embeddings=num_embeddings,
        alpha=alpha,
        reference_embedding=embedding,
    )
    hub.to(device=embedding.weight.device, dtype=embedding.weight.dtype)

    model.embhub = hub
    handle = embedding.register_forward_hook(lambda mod, inp, out: hub(out))
    model._embhub_hook_handle = handle

    if freeze_base:
        for name, param in model.named_parameters():
            if "embhub" not in name:
                param.requires_grad = False

    return hub


def disable_embhub(model: nn.Module) -> None:
    """Set alpha=0 to effectively bypass the hub at inference."""
    if hasattr(model, "embhub"):
        model.embhub.alpha = 0.0


def enable_embhub(model: nn.Module, alpha: float = 0.05) -> None:
    """Re-enable the hub with the given alpha."""
    if hasattr(model, "embhub"):
        model.embhub.alpha = alpha


def remove_embhub(model: nn.Module) -> None:
    """Remove the EmbHub layer entirely, restoring the original model."""
    if hasattr(model, "_embhub_hook_handle"):
        model._embhub_hook_handle.remove()
        del model._embhub_hook_handle
    if hasattr(model, "embhub"):
        del model.embhub


def save_embhub(model: nn.Module, save_directory: str) -> None:
    """Save EmbHub weights and config to a directory."""
    if not hasattr(model, "embhub"):
        raise ValueError("Model does not have an EmbHub layer to save")

    os.makedirs(save_directory, exist_ok=True)

    hub = model.embhub
    state_dict = {k: v.float() for k, v in hub.state_dict().items()}
    torch.save(state_dict, os.path.join(save_directory, EMBHUB_WEIGHTS_NAME))

    config = {
        "num_embeddings": hub.num_embeddings,
        "embedding_dim": hub.embedding_dim,
        "alpha": hub.alpha,
    }
    with open(os.path.join(save_directory, EMBHUB_CONFIG_NAME), "w") as f:
        json.dump(config, f, indent=2)


def load_model_with_embhub(
    model_name_or_path: str,
    num_embeddings: int = 1000,
    alpha: float = 0.05,
    freeze_base: bool = False,
    **model_kwargs,
) -> tuple:
    """Load a pretrained HF causal LM and inject EmbHub.

    If the directory contains saved EmbHub weights (embhub.pt + embhub_config.json),
    loads them. Otherwise injects a freshly initialized EmbHub.

    Returns (model, hub).
    """
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

    config_path = os.path.join(model_name_or_path, EMBHUB_CONFIG_NAME)
    weights_path = os.path.join(model_name_or_path, EMBHUB_WEIGHTS_NAME)

    if os.path.isfile(config_path) and os.path.isfile(weights_path):
        with open(config_path) as f:
            hub_config = json.load(f)
        hub = inject_embhub(
            model,
            num_embeddings=hub_config["num_embeddings"],
            alpha=hub_config["alpha"],
            freeze_base=freeze_base,
        )
        hub.load_state_dict(torch.load(weights_path, map_location=model.device, weights_only=True))
    else:
        hub = inject_embhub(model, num_embeddings=num_embeddings, alpha=alpha, freeze_base=freeze_base)

    return model, hub


def create_model_with_embhub(
    config_name_or_path: str,
    num_embeddings: int = 1000,
    alpha: float = 0.05,
    **config_overrides,
) -> tuple:
    """Create a causal LM from config (random weights) and inject EmbHub.

    Returns (model, hub).
    """
    config = AutoConfig.from_pretrained(config_name_or_path, **config_overrides)
    model = AutoModelForCausalLM.from_config(config)
    hub = inject_embhub(model, num_embeddings=num_embeddings, alpha=alpha)
    return model, hub
