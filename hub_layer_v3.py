"""EmbHub V3 — upgraded anchor block with decoupled keys/values, transform, and gate.

Supports V3 (embedding layer), V4 (multi-head), and V5 (mid-layer) via parameters.
Placement (embedding vs mid-layer) is handled by the wrapper, not this module.

Architecture:
    keys, values = anchor_keys (N x d), anchor_values (N x d)
    w       = softmax(cos(x, keys) * scale)
    mixture = w @ values
    update  = Linear_v(mixture)
    gate    = sigmoid(Linear_g(x))
    output  = x + gate * update

With num_heads > 1 (V4): each head has its own keys/values in a d/h subspace,
does independent cosine selection, then results are concatenated before Linear_v.

Safe init: Linear_v weights = 0, Linear_g bias = gate_bias_init (default -5)
so output ~= x at step 0.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbHubV3(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int = 1000,
        num_heads: int = 1,
        gate_bias_init: float = -5.0,
        reference_weight: torch.Tensor = None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.num_heads = num_heads

        assert embedding_dim % num_heads == 0, \
            f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
        self.head_dim = embedding_dim // num_heads

        self.anchor_keys = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.anchor_values = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

        self.log_logit_scale = nn.Parameter(torch.tensor(math.log(14.0)))

        self.linear_v = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.linear_g = nn.Linear(embedding_dim, embedding_dim)

        self._init_weights(reference_weight, gate_bias_init)

    def _init_weights(self, reference_weight: torch.Tensor = None, gate_bias_init: float = -5.0) -> None:
        if reference_weight is not None:
            std = reference_weight.std().item()
            mean = reference_weight.mean().item()
            self.anchor_keys.data.normal_(mean=mean, std=std)
            self.anchor_values.data.normal_(mean=mean, std=std)
        else:
            nn.init.xavier_uniform_(self.anchor_keys.data)
            nn.init.xavier_uniform_(self.anchor_values.data)

        nn.init.zeros_(self.linear_v.weight)

        nn.init.xavier_uniform_(self.linear_g.weight)
        nn.init.constant_(self.linear_g.bias, gate_bias_init)

    def _select_single_head(self, x: torch.Tensor):
        q = F.normalize(x, dim=-1)
        k = F.normalize(self.anchor_keys, dim=-1)
        scale = self.log_logit_scale.exp().clamp(max=100.0)
        logits = (q @ k.T) * scale
        weights = logits.softmax(dim=-1)
        mixture = weights @ self.anchor_values
        return mixture, weights, logits

    def _select_multi_head(self, x: torch.Tensor):
        B_seq = x.shape[:-1]
        h, d_h = self.num_heads, self.head_dim
        N = self.num_embeddings

        x_heads = x.view(*B_seq, h, d_h)
        keys_heads = self.anchor_keys.view(N, h, d_h)
        values_heads = self.anchor_values.view(N, h, d_h)

        q = F.normalize(x_heads, dim=-1)
        k = F.normalize(keys_heads, dim=-1)

        scale = self.log_logit_scale.exp().clamp(max=100.0)
        logits = torch.einsum("...hd,nhd->...hn", q, k) * scale
        weights = logits.softmax(dim=-1)
        mixture_heads = torch.einsum("...hn,nhd->...hd", weights, values_heads)

        mixture = mixture_heads.reshape(*B_seq, self.embedding_dim)
        return mixture, weights, logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_heads == 1:
            mixture, weights, logits = self._select_single_head(x)
        else:
            mixture, weights, logits = self._select_multi_head(x)

        update = self.linear_v(mixture)
        gate = torch.sigmoid(self.linear_g(x))
        return x + gate * update

    def compute_diagnostics(self, x: torch.Tensor) -> dict:
        with torch.no_grad():
            x_f = x.float()
            scale = self.log_logit_scale.float().exp().clamp(max=100.0)

            if self.num_heads == 1:
                q = F.normalize(x_f, dim=-1)
                k = F.normalize(self.anchor_keys.float(), dim=-1)
                logits = (q @ k.T) * scale
                weights = logits.softmax(dim=-1)
                mixture = weights @ self.anchor_values.float()
            else:
                B_seq = x_f.shape[:-1]
                h, d_h = self.num_heads, self.head_dim
                N = self.num_embeddings
                x_heads = x_f.view(*B_seq, h, d_h)
                keys_f = self.anchor_keys.float().view(N, h, d_h)
                values_f = self.anchor_values.float().view(N, h, d_h)
                q = F.normalize(x_heads, dim=-1)
                k = F.normalize(keys_f, dim=-1)
                logits = torch.einsum("...hd,nhd->...hn", q, k) * scale
                weights = logits.softmax(dim=-1)
                mixture_heads = torch.einsum("...hn,nhd->...hd", weights, values_f)
                mixture = mixture_heads.reshape(*B_seq, self.embedding_dim)

            update = F.linear(mixture, self.linear_v.weight.float())
            gate = torch.sigmoid(F.linear(x_f, self.linear_g.weight.float(), self.linear_g.bias.float()))

            contribution = gate * update

            if self.num_heads == 1:
                entropy = -(weights * weights.clamp(min=1e-12).log()).sum(dim=-1)
                anchor_mass = weights.mean(dim=tuple(range(weights.dim() - 1)))
                max_weight = weights.max(dim=-1).values.mean().item()
            else:
                entropy = -(weights * weights.clamp(min=1e-12).log()).sum(dim=-1)
                entropy = entropy.mean(dim=-1)
                anchor_mass = weights.mean(dim=tuple(range(weights.dim() - 2))).mean(dim=0)
                max_weight = weights.max(dim=-1).values.mean().item()

            uniform_entropy = math.log(self.num_embeddings)

            top10_mass = anchor_mass.topk(min(10, anchor_mass.shape[-1])).values.sum().item()
            uniform_mass = 1.0 / self.num_embeddings
            dead_fraction = (anchor_mass < 0.1 * uniform_mass).float().mean().item()

            contrib_norm = contribution.norm(dim=-1)
            token_norm = x_f.norm(dim=-1)
            norm_ratio = (contrib_norm / token_norm.clamp(min=1e-8)).mean().item()

            gate_mean = gate.mean().item()

            n_sample = min(100, self.num_embeddings)
            sampled_keys = F.normalize(self.anchor_keys[:n_sample].float(), dim=-1)
            pairwise_cos = (sampled_keys @ sampled_keys.T).triu(diagonal=1)
            mask = torch.triu(torch.ones_like(pairwise_cos, dtype=torch.bool), diagonal=1)
            mean_pairwise_cos = pairwise_cos[mask].mean().item()

            return {
                "logit_std": logits.std().item(),
                "entropy_mean": entropy.mean().item(),
                "entropy_std": entropy.std().item(),
                "uniform_entropy": uniform_entropy,
                "effective_anchors": math.exp(entropy.mean().item()),
                "max_weight_mean": max_weight,
                "logit_scale": scale.item(),
                "top10_anchor_mass": top10_mass,
                "dead_anchor_fraction": dead_fraction,
                "norm_ratio": norm_ratio,
                "gate_mean": gate_mean,
                "anchor_pairwise_cosine": mean_pairwise_cos,
            }
