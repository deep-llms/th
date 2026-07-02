import math

import torch
import pytest

from hub_layer_v6 import EmbHubV6


class TestV6Init:

    def test_curriculum_starts_plain(self):
        hub = EmbHubV6(embedding_dim=64, num_embeddings=32, top_k=5)
        hub.train()
        x = torch.randn(2, 10, 64)
        out = hub(x)
        assert torch.equal(out, x), "Step 0 (ramp=0) should output plain x"

    def test_inference_returns_combined(self):
        hub = EmbHubV6(embedding_dim=64, num_embeddings=32, top_k=5)
        hub.eval()
        x = torch.randn(2, 10, 64)
        out = hub(x)
        assert not torch.equal(out, x), "Inference should return combined (not plain x)"

    def test_v6f_inference_returns_combined(self):
        hub = EmbHubV6(embedding_dim=64, num_embeddings=32, top_k=5, use_residual_cap=True)
        hub.eval()
        x = torch.randn(2, 10, 64)
        out = hub(x)
        assert not torch.equal(out, x)

    def test_no_alpha(self):
        hub = EmbHubV6(embedding_dim=64, num_embeddings=32, top_k=5)
        assert not hasattr(hub, "alpha")

    def test_current_step_buffer(self):
        hub = EmbHubV6(embedding_dim=64, num_embeddings=32, top_k=5)
        assert hub.current_step.item() == 0
        hub.current_step.fill_(100)
        assert hub.current_step.item() == 100


class TestV6Shapes:

    def test_output_shape_v6(self):
        hub = EmbHubV6(embedding_dim=64, num_embeddings=32, top_k=5)
        hub.eval()
        assert hub(torch.randn(2, 10, 64)).shape == (2, 10, 64)

    def test_output_shape_v6f(self):
        hub = EmbHubV6(embedding_dim=64, num_embeddings=32, top_k=5, use_residual_cap=True)
        hub.eval()
        assert hub(torch.randn(2, 10, 64)).shape == (2, 10, 64)

    def test_output_shape_training(self):
        hub = EmbHubV6(embedding_dim=64, num_embeddings=32, top_k=5)
        hub.train()
        hub.current_step.fill_(5000)
        assert hub(torch.randn(2, 10, 64)).shape == (2, 10, 64)


class TestV6Curriculum:

    def test_ramp_zero_all_plain(self):
        hub = EmbHubV6(embedding_dim=64, num_embeddings=32, top_k=5, anneal_steps=1000)
        hub.train()
        hub.current_step.fill_(0)
        x = torch.randn(4, 20, 64)
        out = hub(x)
        assert torch.equal(out, x)

    def test_ramp_full_has_stochastic_modes(self):
        hub = EmbHubV6(embedding_dim=64, num_embeddings=32, top_k=5,
                       p_only=0.10, p_both=0.40, anneal_steps=100)
        hub.train()
        hub.current_step.fill_(200)
        x = torch.randn(100, 50, 64)
        torch.manual_seed(42)
        out = hub(x)
        plain_mask = torch.equal(out, x)
        assert not plain_mask, "After full ramp, not all tokens should be plain"

    def test_ramp_increases_with_step(self):
        hub = EmbHubV6(embedding_dim=64, num_embeddings=32, top_k=5, anneal_steps=1000)
        hub.train()
        x = torch.randn(10, 20, 64)
        diffs = []
        for step in [0, 250, 500, 1000, 2000]:
            hub.current_step.fill_(step)
            torch.manual_seed(42)
            out = hub(x)
            diffs.append((out - x).abs().sum().item())
        assert diffs[0] == 0, "Step 0 should be pure plain"
        assert diffs[-1] > diffs[1] > diffs[0], "Stochastic contribution should grow"


class TestV6fResidualCap:

    def test_residual_is_capped(self):
        hub = EmbHubV6(embedding_dim=64, num_embeddings=32, top_k=5,
                       use_residual_cap=True, r_budget=0.3)
        hub.eval()
        x = torch.randn(2, 10, 64) * 10
        concept, _, _, _ = hub._retrieve_concept(x)
        resid = hub._cap_residual(x, concept)
        max_allowed = 0.3 * concept.norm(dim=-1)
        resid_norms = resid.norm(dim=-1)
        assert (resid_norms <= max_allowed + 1e-5).all()

    def test_small_residual_not_stretched(self):
        hub = EmbHubV6(embedding_dim=64, num_embeddings=32, top_k=5,
                       use_residual_cap=True, r_budget=0.3)
        hub.eval()
        x = torch.randn(2, 10, 64) * 0.001
        concept, _, _, _ = hub._retrieve_concept(x)
        resid = hub._cap_residual(x, concept)
        assert torch.allclose(resid, x, atol=1e-6), "Small residual should not be stretched"


class TestV6Gradients:

    def test_gradient_flow_v6(self):
        hub = EmbHubV6(embedding_dim=64, num_embeddings=32, top_k=5)
        hub.train()
        hub.current_step.fill_(5000)
        x = torch.randn(2, 10, 64)
        hub(x).sum().backward()
        assert hub.anchor_keys.grad is not None
        assert hub.anchor_values.grad is not None
        assert hub.log_logit_scale.grad is not None

    def test_gradient_flow_v6f(self):
        hub = EmbHubV6(embedding_dim=64, num_embeddings=32, top_k=5, use_residual_cap=True)
        hub.train()
        hub.current_step.fill_(5000)
        x = torch.randn(2, 10, 64)
        hub(x).sum().backward()
        assert hub.anchor_keys.grad is not None
        assert hub.anchor_values.grad is not None

    def test_all_params_get_gradient(self):
        hub = EmbHubV6(embedding_dim=64, num_embeddings=32, top_k=5)
        hub.train()
        hub.current_step.fill_(5000)
        optimizer = torch.optim.Adam(hub.parameters(), lr=1e-2)
        for _ in range(3):
            optimizer.zero_grad()
            hub(torch.randn(2, 10, 64)).sum().backward()
            optimizer.step()
        optimizer.zero_grad()
        hub(torch.randn(2, 10, 64)).sum().backward()
        for name, p in hub.named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, f"{name} has no gradient"


class TestV6Diagnostics:

    def test_all_keys_present(self):
        hub = EmbHubV6(embedding_dim=64, num_embeddings=32, top_k=5)
        diag = hub.compute_diagnostics(torch.randn(2, 10, 64))
        expected = [
            "logit_std", "entropy_mean", "logit_scale",
            "top10_anchor_mass", "dead_anchor_fraction",
            "topk_mass_total", "norm_ratio", "concept_norm",
            "curriculum_ramp", "anchor_pairwise_cosine",
        ]
        for k in expected:
            assert k in diag, f"Missing key: {k}"

    def test_v6f_has_resid_keys(self):
        hub = EmbHubV6(embedding_dim=64, num_embeddings=32, top_k=5, use_residual_cap=True)
        diag = hub.compute_diagnostics(torch.randn(2, 10, 64))
        assert "resid_norm" in diag
        assert "resid_concept_ratio" in diag

    def test_curriculum_ramp_in_diagnostics(self):
        hub = EmbHubV6(embedding_dim=64, num_embeddings=32, top_k=5, anneal_steps=1000)
        hub.current_step.fill_(500)
        diag = hub.compute_diagnostics(torch.randn(2, 10, 64))
        assert abs(diag["curriculum_ramp"] - 0.5) < 0.01

    def test_diagnostics_bf16(self):
        hub = EmbHubV6(embedding_dim=64, num_embeddings=32, top_k=5).to(torch.bfloat16)
        x = torch.randn(2, 10, 64, dtype=torch.bfloat16)
        diag = hub.compute_diagnostics(x)
        assert isinstance(diag["norm_ratio"], float)

    def test_diagnostics_v6f_bf16(self):
        hub = EmbHubV6(embedding_dim=64, num_embeddings=32, top_k=5, use_residual_cap=True).to(torch.bfloat16)
        x = torch.randn(2, 10, 64, dtype=torch.bfloat16)
        diag = hub.compute_diagnostics(x)
        assert isinstance(diag["resid_norm"], float)


class TestV6StateDict:

    def test_state_dict_keys(self):
        hub = EmbHubV6(embedding_dim=64, num_embeddings=32, top_k=5)
        keys = set(hub.state_dict().keys())
        expected = {"anchor_keys", "anchor_values", "log_logit_scale", "current_step"}
        assert keys == expected

    def test_save_and_load(self):
        hub1 = EmbHubV6(embedding_dim=64, num_embeddings=32, top_k=5, use_residual_cap=True)
        hub1.eval()
        hub1.current_step.fill_(1000)
        state = hub1.state_dict()
        hub2 = EmbHubV6(embedding_dim=64, num_embeddings=32, top_k=5, use_residual_cap=True)
        hub2.load_state_dict(state)
        hub2.eval()
        x = torch.randn(2, 10, 64)
        assert torch.equal(hub1(x), hub2(x))
        assert hub2.current_step.item() == 1000

    def test_param_count(self):
        d, N = 64, 32
        hub = EmbHubV6(embedding_dim=d, num_embeddings=N)
        total = sum(p.numel() for p in hub.parameters())
        expected = (N * d) + (N * d) + 1
        assert total == expected
