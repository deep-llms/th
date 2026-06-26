import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig

from hub_layer import EmbHub, EmbeddingWithHub


def inject_embhub(
    model: nn.Module,
    num_embeddings: int = 1000,
    alpha: float = 0.05,
    freeze_base: bool = False,
) -> EmbHub:
    """Inject an EmbHub layer into any HuggingFace causal LM.

    Returns the EmbHub module so callers can access it directly.
    """
    original_embedding = model.get_input_embeddings()
    if isinstance(original_embedding, EmbeddingWithHub):
        raise ValueError("Model already has an EmbHub layer injected")

    wrapper = EmbeddingWithHub(original_embedding, num_hub_embeddings=num_embeddings, alpha=alpha)
    wrapper.hub.to(device=original_embedding.weight.device, dtype=original_embedding.weight.dtype)
    model.set_input_embeddings(wrapper)

    if freeze_base:
        for name, param in model.named_parameters():
            if "hub" not in name:
                param.requires_grad = False

    return wrapper.hub


def disable_embhub(model: nn.Module) -> None:
    """Set alpha=0 to effectively bypass the hub at inference."""
    emb = model.get_input_embeddings()
    if isinstance(emb, EmbeddingWithHub):
        emb.hub.alpha = 0.0


def enable_embhub(model: nn.Module, alpha: float = 0.05) -> None:
    """Re-enable the hub with the given alpha."""
    emb = model.get_input_embeddings()
    if isinstance(emb, EmbeddingWithHub):
        emb.hub.alpha = alpha


def remove_embhub(model: nn.Module) -> None:
    """Remove the EmbHub wrapper entirely, restoring the original embedding."""
    emb = model.get_input_embeddings()
    if isinstance(emb, EmbeddingWithHub):
        model.set_input_embeddings(emb.original_embedding)


def load_model_with_embhub(
    model_name_or_path: str,
    num_embeddings: int = 1000,
    alpha: float = 0.05,
    freeze_base: bool = False,
    **model_kwargs,
) -> tuple:
    """Load a pretrained HF causal LM and inject EmbHub.

    Returns (model, hub).
    """
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
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
