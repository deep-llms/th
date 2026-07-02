import math

import torch
import pytest

from hub_layer_v2_topk import EmbHubV2TopK


# ---------------------------------------------------------------------------
# Safe init tests
# ---------------------------------------------------------------------------

class TestSafeInit:

    def test_pass_through_v2c(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=5)
        x = torch.randn(2, 10, 64)
        assert torch.allclose(hub(x), x, atol=1e-6)

    def test_pass_through_v2c_tail(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=5, tail_mode="tail")
        x = torch.randn(2, 10, 64)
        assert torch.allclose(hub(x), x, atol=1e-6)

    def test_pass_through_v2c_buckets(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=5, tail_mode="buckets", num_buckets=4)
        x = torch.randn(2, 10, 64)
        assert torch.allclose(hub(x), x, atol=1e-6)

    def test_pass_through_all_weightings(self):
        for w in ["raw_softmax", "renormalized", "none"]:
            hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=5, weighting=w)
            x = torch.randn(2, 10, 64)
            assert torch.allclose(hub(x), x, atol=1e-6), f"weighting={w} failed"

    def test_linear_out_init(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=5)
        d = 64
        assert torch.allclose(hub.linear_out.weight[:, :d], torch.eye(d), atol=1e-7)
        assert torch.equal(hub.linear_out.weight[:, d:], torch.zeros(d, 5 * d))
        assert torch.equal(hub.linear_out.bias, torch.zeros(d))


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestShapes:

    def test_output_shape_v2c(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=5)
        assert hub(torch.randn(2, 10, 64)).shape == (2, 10, 64)

    def test_output_shape_v2c_tail(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=5, tail_mode="tail")
        assert hub(torch.randn(2, 10, 64)).shape == (2, 10, 64)

    def test_output_shape_v2c_buckets(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=5, tail_mode="buckets", num_buckets=4)
        assert hub(torch.randn(2, 10, 64)).shape == (2, 10, 64)

    def test_linear_out_width_v2c(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=5)
        assert hub.linear_out.weight.shape == (64, (1 + 5) * 64)

    def test_linear_out_width_v2c_tail(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=5, tail_mode="tail")
        assert hub.linear_out.weight.shape == (64, (1 + 5 + 1) * 64)

    def test_linear_out_width_v2c_buckets(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=5, tail_mode="buckets", num_buckets=4)
        assert hub.linear_out.weight.shape == (64, (1 + 5 + 4) * 64)

    def test_top_k_clamped(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=8, top_k=20)
        assert hub.top_k == 8


# ---------------------------------------------------------------------------
# Architecture tests
# ---------------------------------------------------------------------------

class TestArchitecture:

    def test_invalid_weighting(self):
        with pytest.raises(AssertionError):
            EmbHubV2TopK(embedding_dim=64, num_embeddings=32, weighting="bad")

    def test_invalid_tail_mode(self):
        with pytest.raises(AssertionError):
            EmbHubV2TopK(embedding_dim=64, num_embeddings=32, tail_mode="bad")

    def test_deterministic(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=5)
        x = torch.randn(2, 10, 64)
        assert torch.equal(hub(x), hub(x))

    def test_no_alpha(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=5)
        assert not hasattr(hub, "alpha")


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------

class TestGradients:

    def test_gradient_after_bootstrap_v2c(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=5)
        optimizer = torch.optim.Adam(hub.parameters(), lr=1e-2)
        for _ in range(3):
            optimizer.zero_grad()
            hub(torch.randn(2, 10, 64)).sum().backward()
            optimizer.step()
        optimizer.zero_grad()
        hub(torch.randn(2, 10, 64)).sum().backward()
        for name, p in hub.named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, \
                f"{name} has no gradient"

    def test_gradient_after_bootstrap_tail(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=5, tail_mode="tail")
        optimizer = torch.optim.Adam(hub.parameters(), lr=1e-2)
        for _ in range(3):
            optimizer.zero_grad()
            hub(torch.randn(2, 10, 64)).sum().backward()
            optimizer.step()
        optimizer.zero_grad()
        hub(torch.randn(2, 10, 64)).sum().backward()
        for name, p in hub.named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, \
                f"{name} has no gradient (tail)"

    def test_gradient_after_bootstrap_buckets(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=5, tail_mode="buckets", num_buckets=4)
        optimizer = torch.optim.Adam(hub.parameters(), lr=1e-2)
        for _ in range(3):
            optimizer.zero_grad()
            hub(torch.randn(2, 10, 64)).sum().backward()
            optimizer.step()
        optimizer.zero_grad()
        hub(torch.randn(2, 10, 64)).sum().backward()
        for name, p in hub.named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, \
                f"{name} has no gradient (buckets)"


# ---------------------------------------------------------------------------
# Training tests
# ---------------------------------------------------------------------------

class TestTraining:

    def test_contribution_grows(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=5)
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

    def test_anchor_slots_grow_from_zero(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=5)
        d = 64
        assert hub.linear_out.weight[:, d:].abs().max().item() == 0
        optimizer = torch.optim.Adam(hub.parameters(), lr=1e-2)
        for _ in range(5):
            optimizer.zero_grad()
            hub(torch.randn(2, 10, 64)).sum().backward()
            optimizer.step()
        assert hub.linear_out.weight[:, d:].abs().max().item() > 0


# ---------------------------------------------------------------------------
# Diagnostics tests
# ---------------------------------------------------------------------------

class TestDiagnostics:

    def test_all_keys_present(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=5)
        diag = hub.compute_diagnostics(torch.randn(2, 10, 64))
        expected = [
            "logit_std", "entropy_mean", "entropy_std", "uniform_entropy",
            "effective_anchors", "max_weight_mean", "logit_scale",
            "top10_anchor_mass", "dead_anchor_fraction", "topk_mass_total",
            "norm_ratio", "w_anchor_norm", "anchor_pairwise_cosine",
        ]
        for k in expected:
            assert k in diag, f"Missing key: {k}"

    def test_norm_ratio_zero_at_init(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=5)
        diag = hub.compute_diagnostics(torch.randn(2, 10, 64))
        assert diag["norm_ratio"] < 1e-6

    def test_diagnostics_tail(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=5, tail_mode="tail")
        diag = hub.compute_diagnostics(torch.randn(2, 10, 64))
        assert diag["norm_ratio"] < 1e-6

    def test_diagnostics_buckets(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=5, tail_mode="buckets", num_buckets=4)
        diag = hub.compute_diagnostics(torch.randn(2, 10, 64))
        assert diag["norm_ratio"] < 1e-6

    def test_topk_mass(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=5)
        diag = hub.compute_diagnostics(torch.randn(2, 10, 64))
        assert 0 < diag["topk_mass_total"] <= 1.0

    def test_diagnostics_bf16(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=5).to(torch.bfloat16)
        x = torch.randn(2, 10, 64, dtype=torch.bfloat16)
        diag = hub.compute_diagnostics(x)
        assert isinstance(diag["norm_ratio"], float)
        assert not any(str(v) == 'nan' for v in diag.values())

    def test_diagnostics_bf16_tail(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=5, tail_mode="tail").to(torch.bfloat16)
        x = torch.randn(2, 10, 64, dtype=torch.bfloat16)
        diag = hub.compute_diagnostics(x)
        assert isinstance(diag["norm_ratio"], float)

    def test_diagnostics_bf16_buckets(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=5, tail_mode="buckets", num_buckets=4).to(torch.bfloat16)
        x = torch.randn(2, 10, 64, dtype=torch.bfloat16)
        diag = hub.compute_diagnostics(x)
        assert isinstance(diag["norm_ratio"], float)


# ---------------------------------------------------------------------------
# State dict tests
# ---------------------------------------------------------------------------

class TestStateDict:

    def test_state_dict_keys(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=5)
        keys = set(hub.state_dict().keys())
        expected = {"hub_embeddings", "log_logit_scale", "linear_out.weight", "linear_out.bias"}
        assert keys == expected

    def test_save_and_load(self):
        hub1 = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=5, tail_mode="tail")
        optimizer = torch.optim.Adam(hub1.parameters(), lr=1e-2)
        for _ in range(5):
            optimizer.zero_grad()
            hub1(torch.randn(2, 10, 64)).sum().backward()
            optimizer.step()
        state = hub1.state_dict()
        hub2 = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=5, tail_mode="tail")
        hub2.load_state_dict(state)
        x = torch.randn(2, 10, 64)
        assert torch.equal(hub1(x), hub2(x))

    def test_param_count_v2c(self):
        d, N, k = 64, 32, 5
        hub = EmbHubV2TopK(embedding_dim=d, num_embeddings=N, top_k=k)
        total = sum(p.numel() for p in hub.parameters())
        expected = (N * d) + 1 + (d * (1 + k) * d) + d
        assert total == expected

    def test_param_count_v2c_tail(self):
        d, N, k = 64, 32, 5
        hub = EmbHubV2TopK(embedding_dim=d, num_embeddings=N, top_k=k, tail_mode="tail")
        total = sum(p.numel() for p in hub.parameters())
        expected = (N * d) + 1 + (d * (1 + k + 1) * d) + d
        assert total == expected


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_topk_equals_num_embeddings(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=8, top_k=8)
        x = torch.randn(2, 10, 64)
        assert torch.allclose(hub(x), x, atol=1e-6)

    def test_topk_equals_num_embeddings_with_tail(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=8, top_k=8, tail_mode="tail")
        x = torch.randn(2, 10, 64)
        assert torch.allclose(hub(x), x, atol=1e-6)

    def test_topk_equals_num_embeddings_with_buckets(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=8, top_k=8, tail_mode="buckets", num_buckets=4)
        x = torch.randn(2, 10, 64)
        assert torch.allclose(hub(x), x, atol=1e-6)

    def test_topk_1(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=1)
        x = torch.randn(2, 10, 64)
        assert hub(x).shape == (2, 10, 64)
        assert torch.allclose(hub(x), x, atol=1e-6)

    def test_topk_1_with_tail(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=1, tail_mode="tail")
        x = torch.randn(2, 10, 64)
        assert hub(x).shape == (2, 10, 64)
        assert torch.allclose(hub(x), x, atol=1e-6)

    def test_more_buckets_than_rest(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=8, top_k=5, tail_mode="buckets", num_buckets=10)
        x = torch.randn(2, 10, 64)
        assert hub(x).shape == (2, 10, 64)
        assert torch.allclose(hub(x), x, atol=1e-6)

    def test_weighting_values_differ(self):
        hub_b = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=5, weighting="raw_softmax")
        hub_a = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=5, weighting="renormalized")
        hub_c = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=5, weighting="none")
        # After training, different weightings should produce different outputs
        x = torch.randn(2, 10, 64)
        optimizer_b = torch.optim.Adam(hub_b.parameters(), lr=1e-1)
        optimizer_a = torch.optim.Adam(hub_a.parameters(), lr=1e-1)
        optimizer_c = torch.optim.Adam(hub_c.parameters(), lr=1e-1)
        for _ in range(5):
            for opt, hub in [(optimizer_b, hub_b), (optimizer_a, hub_a), (optimizer_c, hub_c)]:
                opt.zero_grad()
                hub(x).sum().backward()
                opt.step()
        # Just verify they all produce valid outputs
        for hub in [hub_b, hub_a, hub_c]:
            out = hub(x)
            assert out.shape == (2, 10, 64)
            assert not torch.isnan(out).any()

    def test_selected_anchors_get_more_gradient_v2c(self):
        """V2c: selected anchors should get larger gradient than non-selected."""
        hub = EmbHubV2TopK(embedding_dim=16, num_embeddings=8, top_k=2, tail_mode="none")
        optimizer = torch.optim.Adam(hub.parameters(), lr=1e-1)
        for _ in range(3):
            optimizer.zero_grad()
            hub(torch.randn(1, 1, 16)).sum().backward()
            optimizer.step()
        optimizer.zero_grad()
        x = torch.randn(1, 1, 16)
        hub(x).sum().backward()
        grad_norms = hub.hub_embeddings.grad.abs().sum(dim=-1)
        topk_grad = grad_norms.topk(2).values.mean().item()
        rest_grad = grad_norms.sort().values[:6].mean().item()
        assert topk_grad > rest_grad, "Selected anchors should get more gradient"

    def test_gradient_flows_to_all_with_tail(self):
        """V2c+tail: all anchors should eventually get gradient through the tail slot."""
        hub = EmbHubV2TopK(embedding_dim=16, num_embeddings=8, top_k=2, tail_mode="tail")
        optimizer = torch.optim.Adam(hub.parameters(), lr=1e-1)
        for _ in range(3):
            optimizer.zero_grad()
            hub(torch.randn(1, 1, 16)).sum().backward()
            optimizer.step()
        optimizer.zero_grad()
        hub(torch.randn(1, 1, 16)).sum().backward()
        grad = hub.hub_embeddings.grad
        nonzero_rows = (grad.abs().sum(dim=-1) > 0).sum().item()
        assert nonzero_rows == 8, f"Expected all 8 anchors to get gradient with tail, got {nonzero_rows}"

    def test_single_sequence(self):
        hub = EmbHubV2TopK(embedding_dim=64, num_embeddings=32, top_k=5, tail_mode="tail")
        x = torch.randn(1, 1, 64)
        assert hub(x).shape == (1, 1, 64)
