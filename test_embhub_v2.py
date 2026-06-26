import torch
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from hub_layer_v2 import EmbHub
from model_wrapper_v2 import (
    inject_embhub,
    disable_embhub,
    enable_embhub,
    remove_embhub,
    save_embhub,
    load_model_with_embhub,
)

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def dummy_input(tokenizer):
    return tokenizer("The quick brown fox jumps over the lazy dog", return_tensors="pt")


# ---------------------------------------------------------------------------
# EmbHub layer unit tests
# ---------------------------------------------------------------------------

class TestEmbHubLayer:

    def test_output_shape(self):
        hub = EmbHub(embedding_dim=64, num_embeddings=32, alpha=0.1)
        x = torch.randn(2, 10, 64)
        assert hub(x).shape == x.shape

    def test_alpha_zero_is_identity(self):
        hub = EmbHub(embedding_dim=64, num_embeddings=32, alpha=0.0)
        x = torch.randn(2, 10, 64)
        assert torch.equal(hub(x), x)

    def test_alpha_nonzero_changes_output(self):
        hub = EmbHub(embedding_dim=64, num_embeddings=32, alpha=0.1)
        x = torch.randn(2, 10, 64)
        assert not torch.equal(hub(x), x)

    def test_standalone_defaults(self):
        hub = EmbHub(embedding_dim=64, num_embeddings=32)
        assert hub.hub_embeddings.dtype == torch.float32
        assert hub.hub_embeddings.device.type == "cpu"
        assert hub(torch.randn(2, 5, 64)).shape == (2, 5, 64)

    def test_gradient_flows_to_hub_embeddings(self):
        hub = EmbHub(embedding_dim=64, num_embeddings=32, alpha=0.1)
        x = torch.randn(2, 10, 64)
        out = hub(x)
        loss = out.sum()
        loss.backward()
        assert hub.hub_embeddings.grad is not None
        assert hub.hub_embeddings.grad.shape == hub.hub_embeddings.shape
        assert hub.hub_embeddings.grad.abs().sum() > 0

    def test_deterministic_output(self):
        hub = EmbHub(embedding_dim=64, num_embeddings=32, alpha=0.1)
        x = torch.randn(2, 10, 64)
        out1 = hub(x)
        out2 = hub(x)
        assert torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# Model integration tests
# ---------------------------------------------------------------------------

class TestModelIntegration:

    def test_forward_pass_shape(self, dummy_input):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        input_ids = dummy_input["input_ids"]

        original_output = model(input_ids=input_ids)
        inject_embhub(model, num_embeddings=64, alpha=0.05)
        hub_output = model(input_ids=input_ids)

        assert hub_output.logits.shape == original_output.logits.shape

    def test_embhub_changes_output(self, dummy_input):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        input_ids = dummy_input["input_ids"]

        original_output = model(input_ids=input_ids)
        inject_embhub(model, num_embeddings=64, alpha=0.05)
        hub_output = model(input_ids=input_ids)

        assert not torch.equal(hub_output.logits, original_output.logits)

    def test_disable_matches_original(self, dummy_input):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        input_ids = dummy_input["input_ids"]

        original_output = model(input_ids=input_ids)
        inject_embhub(model, num_embeddings=64, alpha=0.05)
        disable_embhub(model)
        disabled_output = model(input_ids=input_ids)

        assert torch.allclose(disabled_output.logits, original_output.logits, atol=1e-6)

    def test_enable_after_disable(self, dummy_input):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        input_ids = dummy_input["input_ids"]

        inject_embhub(model, num_embeddings=64, alpha=0.05)
        enabled_output = model(input_ids=input_ids)

        disable_embhub(model)
        disabled_output = model(input_ids=input_ids)

        enable_embhub(model, alpha=0.05)
        reenabled_output = model(input_ids=input_ids)

        assert not torch.equal(enabled_output.logits, disabled_output.logits)
        assert torch.equal(enabled_output.logits, reenabled_output.logits)

    def test_remove_restores_original(self, dummy_input):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        input_ids = dummy_input["input_ids"]

        original_output = model(input_ids=input_ids)
        inject_embhub(model, num_embeddings=64, alpha=0.05)
        remove_embhub(model)
        restored_output = model(input_ids=input_ids)

        assert torch.allclose(restored_output.logits, original_output.logits, atol=1e-6)
        assert not hasattr(model, "embhub")
        assert not hasattr(model, "_embhub_hook_handle")

    def test_freeze_base(self):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        inject_embhub(model, num_embeddings=64, alpha=0.05, freeze_base=True)

        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert len(trainable) == 2
        assert any("hub_embeddings" in n for n in trainable)
        assert any("log_logit_scale" in n for n in trainable)

    def test_double_inject_raises(self):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        inject_embhub(model, num_embeddings=64)
        with pytest.raises(ValueError, match="already has an EmbHub"):
            inject_embhub(model, num_embeddings=64)

    def test_dtype_matches_model(self):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
        hub = inject_embhub(model, num_embeddings=64)
        assert hub.hub_embeddings.dtype == torch.float16

    def test_state_dict_keys_preserved(self):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        original_keys = set(model.state_dict().keys())

        inject_embhub(model, num_embeddings=64)
        new_keys = set(model.state_dict().keys())

        added_keys = new_keys - original_keys
        removed_keys = original_keys - new_keys

        assert removed_keys == set(), f"Keys were renamed or removed: {removed_keys}"
        assert added_keys == {"embhub.hub_embeddings", "embhub.log_logit_scale"}, f"Unexpected new keys: {added_keys}"


# ---------------------------------------------------------------------------
# Training correctness tests
# ---------------------------------------------------------------------------

class TestTraining:

    def test_backward_pass_gradient_flow(self, dummy_input):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        hub = inject_embhub(model, num_embeddings=64, alpha=0.05, freeze_base=True)

        input_ids = dummy_input["input_ids"]
        labels = input_ids.clone()

        output = model(input_ids=input_ids, labels=labels)
        output.loss.backward()

        assert hub.hub_embeddings.grad is not None
        assert hub.hub_embeddings.grad.abs().sum() > 0

    def test_optimizer_step_updates_hub(self, dummy_input):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        hub = inject_embhub(model, num_embeddings=64, alpha=0.05, freeze_base=True)

        input_ids = dummy_input["input_ids"]
        labels = input_ids.clone()
        optimizer = torch.optim.Adam([hub.hub_embeddings], lr=1e-3)

        weights_before = hub.hub_embeddings.data.clone()

        output = model(input_ids=input_ids, labels=labels)
        output.loss.backward()
        optimizer.step()

        assert not torch.equal(hub.hub_embeddings.data, weights_before)

    def test_frozen_base_not_updated(self, dummy_input):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        inject_embhub(model, num_embeddings=64, alpha=0.05, freeze_base=True)

        base_weights_before = {
            n: p.data.clone() for n, p in model.named_parameters() if "embhub" not in n
        }

        input_ids = dummy_input["input_ids"]
        labels = input_ids.clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        output = model(input_ids=input_ids, labels=labels)
        output.loss.backward()
        optimizer.step()

        for name, param in model.named_parameters():
            if "embhub" not in name:
                assert torch.equal(param.data, base_weights_before[name]), \
                    f"Frozen param {name} was modified"

    def test_training_step_reduces_loss(self, dummy_input):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        hub = inject_embhub(model, num_embeddings=64, alpha=0.05, freeze_base=True)

        input_ids = dummy_input["input_ids"]
        labels = input_ids.clone()
        optimizer = torch.optim.Adam([hub.hub_embeddings], lr=1e-2)

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            output = model(input_ids=input_ids, labels=labels)
            output.loss.backward()
            optimizer.step()
            losses.append(output.loss.item())

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"

    def test_reproducibility(self):
        model1 = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        model2 = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)

        torch.manual_seed(42)
        hub1 = inject_embhub(model1, num_embeddings=64, alpha=0.05)

        torch.manual_seed(42)
        hub2 = inject_embhub(model2, num_embeddings=64, alpha=0.05)

        assert torch.equal(hub1.hub_embeddings.data, hub2.hub_embeddings.data)


# ---------------------------------------------------------------------------
# Save / load tests
# ---------------------------------------------------------------------------

class TestSaveLoad:

    def test_save_and_load_weights(self, dummy_input, tmp_path):
        save_dir = str(tmp_path / "checkpoint")

        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        inject_embhub(model, num_embeddings=64, alpha=0.07)
        input_ids = dummy_input["input_ids"]
        original_output = model(input_ids=input_ids)
        original_hub_weights = model.embhub.hub_embeddings.data.clone()

        model.save_pretrained(save_dir)
        save_embhub(model, save_dir)

        loaded_model, loaded_hub = load_model_with_embhub(save_dir, torch_dtype=torch.float32)
        loaded_output = loaded_model(input_ids=input_ids)

        assert loaded_hub.alpha == 0.07
        assert loaded_hub.num_embeddings == 64
        assert torch.equal(loaded_hub.hub_embeddings.data, original_hub_weights)
        assert torch.allclose(loaded_output.logits, original_output.logits, atol=1e-6)

    def test_save_and_load_preserves_base_weights(self, tmp_path):
        save_dir = str(tmp_path / "checkpoint")

        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        base_weights = {
            n: p.data.clone() for n, p in model.named_parameters()
        }

        inject_embhub(model, num_embeddings=64)
        model.save_pretrained(save_dir)
        save_embhub(model, save_dir)

        loaded_model, _ = load_model_with_embhub(save_dir, torch_dtype=torch.float32)

        for name, param in loaded_model.named_parameters():
            if "embhub" not in name:
                assert torch.equal(param.data, base_weights[name]), \
                    f"Base weight {name} changed after save/load"

    def test_save_and_load_after_training(self, dummy_input, tmp_path):
        save_dir = str(tmp_path / "trained_checkpoint")

        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        hub = inject_embhub(model, num_embeddings=64, alpha=0.05, freeze_base=True)

        input_ids = dummy_input["input_ids"]
        labels = input_ids.clone()
        optimizer = torch.optim.Adam([hub.hub_embeddings], lr=1e-2)

        for _ in range(3):
            optimizer.zero_grad()
            output = model(input_ids=input_ids, labels=labels)
            output.loss.backward()
            optimizer.step()

        trained_output = model(input_ids=input_ids)
        trained_hub_weights = hub.hub_embeddings.data.clone()

        model.save_pretrained(save_dir)
        save_embhub(model, save_dir)

        loaded_model, loaded_hub = load_model_with_embhub(
            save_dir, freeze_base=True, torch_dtype=torch.float32,
        )
        loaded_output = loaded_model(input_ids=input_ids)

        assert torch.equal(loaded_hub.hub_embeddings.data, trained_hub_weights)
        assert torch.allclose(loaded_output.logits, trained_output.logits, atol=1e-6)

    def test_load_without_embhub_files(self, tmp_path):
        save_dir = str(tmp_path / "base_checkpoint")

        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        model.save_pretrained(save_dir)

        loaded_model, loaded_hub = load_model_with_embhub(
            save_dir, num_embeddings=32, alpha=0.1, torch_dtype=torch.float32,
        )

        assert loaded_hub.num_embeddings == 32
        assert loaded_hub.alpha == 0.1
        assert hasattr(loaded_model, "embhub")

    def test_save_without_embhub_raises(self):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        with pytest.raises(ValueError, match="does not have an EmbHub"):
            save_embhub(model, "/tmp/should_not_exist")

    def test_save_and_load_float16(self, dummy_input, tmp_path):
        save_dir = str(tmp_path / "fp16_checkpoint")

        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
        inject_embhub(model, num_embeddings=64, alpha=0.05)
        original_hub_weights = model.embhub.hub_embeddings.data.clone()

        model.save_pretrained(save_dir)
        save_embhub(model, save_dir)

        loaded_model, loaded_hub = load_model_with_embhub(save_dir, torch_dtype=torch.float16)

        assert loaded_hub.hub_embeddings.dtype == torch.float16
        assert torch.equal(loaded_hub.hub_embeddings.data, original_hub_weights)
