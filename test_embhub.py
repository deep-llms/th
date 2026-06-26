import torch
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from hub_layer import EmbHub, EmbeddingWithHub
from model_wrapper import (
    inject_embhub,
    disable_embhub,
    enable_embhub,
    remove_embhub,
)

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"


@pytest.fixture(scope="module")
def base_model():
    return AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def dummy_input(tokenizer):
    return tokenizer("The quick brown fox jumps over the lazy dog", return_tensors="pt")


class TestEmbHubLayer:

    def test_output_shape(self):
        hub = EmbHub(embedding_dim=64, num_embeddings=32, alpha=0.1)
        torch.nn.init.normal_(hub.hub_embeddings)
        x = torch.randn(2, 10, 64)
        out = hub(x)
        assert out.shape == x.shape

    def test_alpha_zero_is_identity(self):
        hub = EmbHub(embedding_dim=64, num_embeddings=32, alpha=0.0)
        torch.nn.init.normal_(hub.hub_embeddings)
        x = torch.randn(2, 10, 64)
        out = hub(x)
        assert torch.equal(out, x)

    def test_alpha_nonzero_changes_output(self):
        hub = EmbHub(embedding_dim=64, num_embeddings=32, alpha=0.1)
        torch.nn.init.normal_(hub.hub_embeddings)
        x = torch.randn(2, 10, 64)
        out = hub(x)
        assert not torch.equal(out, x)


class TestModelIntegration:

    def test_forward_pass_shape(self, base_model, dummy_input):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        input_ids = dummy_input["input_ids"]

        original_output = model(input_ids=input_ids)
        inject_embhub(model, num_embeddings=64, alpha=0.05)
        hub_output = model(input_ids=input_ids)

        assert hub_output.logits.shape == original_output.logits.shape

    def test_disable_matches_original(self, base_model, dummy_input):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        input_ids = dummy_input["input_ids"]

        original_output = model(input_ids=input_ids)
        inject_embhub(model, num_embeddings=64, alpha=0.05)
        disable_embhub(model)
        disabled_output = model(input_ids=input_ids)

        assert torch.allclose(disabled_output.logits, original_output.logits, atol=1e-6)

    def test_remove_restores_original(self, base_model, dummy_input):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        input_ids = dummy_input["input_ids"]

        original_output = model(input_ids=input_ids)
        inject_embhub(model, num_embeddings=64, alpha=0.05)
        remove_embhub(model)
        restored_output = model(input_ids=input_ids)

        assert torch.allclose(restored_output.logits, original_output.logits, atol=1e-6)

    def test_freeze_base(self):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        inject_embhub(model, num_embeddings=64, alpha=0.05, freeze_base=True)

        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert len(trainable) == 1
        assert "hub_embeddings" in trainable[0]

    def test_double_inject_raises(self):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        inject_embhub(model, num_embeddings=64)
        with pytest.raises(ValueError, match="already has an EmbHub"):
            inject_embhub(model, num_embeddings=64)

    def test_dtype_matches_model(self):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
        hub = inject_embhub(model, num_embeddings=64)
        assert hub.hub_embeddings.dtype == torch.float16

    def test_standalone_embhub_defaults(self):
        hub = EmbHub(embedding_dim=64, num_embeddings=32)
        assert hub.hub_embeddings.dtype == torch.float32
        assert hub.hub_embeddings.device.type == "cpu"
        out = hub(torch.randn(2, 5, 64))
        assert out.shape == (2, 5, 64)
