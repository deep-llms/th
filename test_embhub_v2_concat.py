import math

import torch
import pytest

from hub_layer_v2_concat import EmbHubV2Concat


# ---------------------------------------------------------------------------
# Safe init tests
# ---------------------------------------------------------------------------

class TestSafeInit:

    def test_pass_through_v2(self):
        hub = EmbHubV2Concat(embedding_dim=64, num_embeddings=32, use_mlp=False)
        x = torch.randn(2, 10, 64)
        out = hub(x)
        assert torch.allclose(out, x, atol=1e-6)

    def test_pass_through_v2b(self):
        hub = EmbHubV2Concat(embedding_dim=64, num_embeddings=32, use_mlp=True)
        x = torch.randn(2, 10, 64)
        out = hub(x)
        assert torch.allclose(out, x, atol=1e-6)

    def test_linear_out_init_identity_zero(self):
        hub = EmbHubV2Concat(embedding_dim=64, num_embeddings=32)
        d = 64
        assert torch.allclose(hub.linear_out.weight[:, :d], torch.eye(d), atol=1e-7)
        assert torch.equal(hub.linear_out.weight[:, d:], torch.zeros(d, d))
        assert torch.equal(hub.linear_out.bias, torch.zeros(d))

    def test_no_alpha(self):
        hub = EmbHubV2Concat(embedding_dim=64, num_embeddings=32)
        assert not hasattr(hub, "alpha")


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestShapes:

    def test_output_shape(self):
        hub = EmbHubV2Concat(embedding_dim=64, num_embeddings=32)
        x = torch.randn(2, 10, 64)
        assert hub(x).shape == (2, 10, 64)

    def test_output_shape_v2b(self):
        hub = EmbHubV2Concat(embedding_dim=64, num_embeddings=32, use_mlp=True)
        x = torch.randn(2, 10, 64)
        assert hub(x).shape == (2, 10, 64)

    def test_various_batch_sizes(self):
        hub = EmbHubV2Concat(embedding_dim=64, num_embeddings=32)
        for shape in [(1, 5, 64), (4, 20, 64), (8, 100, 64)]:
            assert hub(torch.randn(*shape)).shape == shape


# ---------------------------------------------------------------------------
# Architecture tests
# ---------------------------------------------------------------------------

class TestArchitecture:

    def test_shared_keys_values(self):
        hub = EmbHubV2Concat(embedding_dim=64, num_embeddings=32)
        assert hasattr(hub, "hub_embeddings")
        assert not hasattr(hub, "anchor_keys")
        assert not hasattr(hub, "anchor_values")

    def test_linear_out_shape(self):
        hub = EmbHubV2Concat(embedding_dim=64, num_embeddings=32)
        assert hub.linear_out.weight.shape == (64, 128)
        assert hub.linear_out.bias.shape == (64,)

    def test_v2_no_mlp(self):
        hub = EmbHubV2Concat(embedding_dim=64, num_embeddings=32, use_mlp=False)
        assert not hasattr(hub, "mlp")

    def test_v2b_has_mlp(self):
        hub = EmbHubV2Concat(embedding_dim=64, num_embeddings=32, use_mlp=True)
        assert hasattr(hub, "mlp")
        assert isinstance(hub.mlp[0], torch.nn.Linear)
        assert isinstance(hub.mlp[1], torch.nn.GELU)

    def test_deterministic(self):
        hub = EmbHubV2Concat(embedding_dim=64, num_embeddings=32)
        x = torch.randn(2, 10, 64)
        assert torch.equal(hub(x), hub(x))

    def test_reference_weight_init(self):
        ref = torch.randn(1000, 64) * 0.02
        hub = EmbHubV2Concat(embedding_dim=64, num_embeddings=32, reference_weight=ref)
        assert abs(hub.hub_embeddings.std().item() - ref.std().item()) < 0.01


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------

class TestGradients:

    def test_gradient_flow_v2(self):
        hub = EmbHubV2Concat(embedding_dim=64, num_embeddings=32, use_mlp=False)
        x = torch.randn(2, 10, 64)
        hub(x).sum().backward()
        assert hub.hub_embeddings.grad is not None
        assert hub.linear_out.weight.grad is not None
        assert hub.log_logit_scale.grad is not None

    def test_gradient_flow_v2b(self):
        hub = EmbHubV2Concat(embedding_dim=64, num_embeddings=32, use_mlp=True)
        x = torch.randn(2, 10, 64)
        hub(x).sum().backward()
        assert hub.hub_embeddings.grad is not None
        assert hub.linear_out.weight.grad is not None
        assert hub.mlp[0].weight.grad is not None

    def test_all_params_get_gradient_after_training(self):
        hub = EmbHubV2Concat(embedding_dim=64, num_embeddings=32, use_mlp=True)
        optimizer = torch.optim.Adam(hub.parameters(), lr=1e-3)
        for _ in range(3):
            optimizer.zero_grad()
            hub(torch.randn(2, 10, 64)).sum().backward()
            optimizer.step()
        optimizer.zero_grad()
        hub(torch.randn(2, 10, 64)).sum().backward()
        for name, p in hub.named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, \
                f"{name} has no gradient"


# ---------------------------------------------------------------------------
# Training tests
# ---------------------------------------------------------------------------

class TestTraining:

    def test_w_mix_grows_from_zero(self):
        hub = EmbHubV2Concat(embedding_dim=64, num_embeddings=32)
        d = 64
        assert hub.linear_out.weight[:, d:].abs().max().item() == 0
        optimizer = torch.optim.Adam(hub.parameters(), lr=1e-2)
        for _ in range(5):
            optimizer.zero_grad()
            hub(torch.randn(2, 10, 64)).sum().backward()
            optimizer.step()
        assert hub.linear_out.weight[:, d:].abs().max().item() > 0

    def test_contribution_grows(self):
        hub = EmbHubV2Concat(embedding_dim=64, num_embeddings=32)
        optimizer = torch.optim.Adam(hub.parameters(), lr=1e-2)
        x = torch.randn(2, 10, 64)
        norms = []
        for _ in range(10):
            optimizer.zero_grad()
            out = hub(x)
            norms.append((out - x).norm().item())
            out.sum().backward()
            optimizer.step()
        assert norms[0] < 1e-4
        assert norms[-1] > norms[0]


# ---------------------------------------------------------------------------
# Diagnostics tests
# ---------------------------------------------------------------------------

class TestDiagnostics:

    def test_all_keys_present(self):
        hub = EmbHubV2Concat(embedding_dim=64, num_embeddings=32)
        diag = hub.compute_diagnostics(torch.randn(2, 10, 64))
        expected = [
            "logit_std", "entropy_mean", "entropy_std", "uniform_entropy",
            "effective_anchors", "max_weight_mean", "logit_scale",
            "top10_anchor_mass", "dead_anchor_fraction", "norm_ratio",
            "w_mix_norm", "anchor_pairwise_cosine",
        ]
        for k in expected:
            assert k in diag, f"Missing key: {k}"

    def test_norm_ratio_zero_at_init(self):
        hub = EmbHubV2Concat(embedding_dim=64, num_embeddings=32)
        diag = hub.compute_diagnostics(torch.randn(2, 10, 64))
        assert diag["norm_ratio"] < 1e-6

    def test_w_mix_norm_zero_at_init(self):
        hub = EmbHubV2Concat(embedding_dim=64, num_embeddings=32)
        diag = hub.compute_diagnostics(torch.randn(2, 10, 64))
        assert diag["w_mix_norm"] < 1e-6

    def test_logit_scale_at_init(self):
        hub = EmbHubV2Concat(embedding_dim=64, num_embeddings=32)
        diag = hub.compute_diagnostics(torch.randn(2, 10, 64))
        assert abs(diag["logit_scale"] - 14.0) < 0.01

    def test_diagnostics_v2b(self):
        hub = EmbHubV2Concat(embedding_dim=64, num_embeddings=32, use_mlp=True)
        diag = hub.compute_diagnostics(torch.randn(2, 10, 64))
        assert diag["norm_ratio"] < 1e-6

    def test_diagnostics_bf16(self):
        hub = EmbHubV2Concat(embedding_dim=64, num_embeddings=32).to(torch.bfloat16)
        x = torch.randn(2, 10, 64, dtype=torch.bfloat16)
        diag = hub.compute_diagnostics(x)
        assert isinstance(diag["norm_ratio"], float)

    def test_diagnostics_v2b_bf16(self):
        hub = EmbHubV2Concat(embedding_dim=64, num_embeddings=32, use_mlp=True).to(torch.bfloat16)
        x = torch.randn(2, 10, 64, dtype=torch.bfloat16)
        diag = hub.compute_diagnostics(x)
        assert isinstance(diag["norm_ratio"], float)
        assert hub.mlp[0].weight.dtype == torch.bfloat16, \
            "compute_diagnostics must not mutate MLP dtype"


# ---------------------------------------------------------------------------
# State dict tests
# ---------------------------------------------------------------------------

class TestStateDict:

    def test_state_dict_keys_v2(self):
        hub = EmbHubV2Concat(embedding_dim=64, num_embeddings=32, use_mlp=False)
        keys = set(hub.state_dict().keys())
        expected = {"hub_embeddings", "log_logit_scale", "linear_out.weight", "linear_out.bias"}
        assert keys == expected

    def test_state_dict_keys_v2b(self):
        hub = EmbHubV2Concat(embedding_dim=64, num_embeddings=32, use_mlp=True)
        keys = set(hub.state_dict().keys())
        expected = {
            "hub_embeddings", "log_logit_scale",
            "linear_out.weight", "linear_out.bias",
            "mlp.0.weight", "mlp.0.bias",
        }
        assert keys == expected

    def test_save_and_load(self):
        hub1 = EmbHubV2Concat(embedding_dim=64, num_embeddings=32, use_mlp=True)
        optimizer = torch.optim.Adam(hub1.parameters(), lr=1e-2)
        for _ in range(5):
            optimizer.zero_grad()
            hub1(torch.randn(2, 10, 64)).sum().backward()
            optimizer.step()
        state = hub1.state_dict()
        hub2 = EmbHubV2Concat(embedding_dim=64, num_embeddings=32, use_mlp=True)
        hub2.load_state_dict(state)
        x = torch.randn(2, 10, 64)
        assert torch.equal(hub1(x), hub2(x))

    def test_param_count_v2(self):
        d, N = 64, 32
        hub = EmbHubV2Concat(embedding_dim=d, num_embeddings=N, use_mlp=False)
        total = sum(p.numel() for p in hub.parameters())
        expected = (N * d) + 1 + (d * 2 * d) + d
        assert total == expected

    def test_param_count_v2b(self):
        d, N = 64, 32
        hub = EmbHubV2Concat(embedding_dim=d, num_embeddings=N, use_mlp=True)
        total = sum(p.numel() for p in hub.parameters())
        expected = (N * d) + 1 + (d * 2 * d) + d + (d * d) + d
        assert total == expected
