"""EmbHub V2-concat — concat + linear combination (V2 and V2b).

V2:  output = Linear([x ; mixture])
V2b: output = Linear_out([x ; GELU(Linear_v(mixture))])

Both use the same cosine + learnable-temperature selection as V2/V3.
Keys and values are the SAME vectors (not decoupled like V3).

Safe init: Linear weight = [Identity | 0], bias = 0.
So output = I*x + 0*mixture = x at step 0.

Placement (embedding vs mid-layer) is handled by model_wrapper_v3.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbHubV2Concat(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int = 1000,
        use_mlp: bool = False,
        reference_weight: torch.Tensor = None,
    ):
        """
        Args:
            embedding_dim: dimension of input vectors
            num_embeddings: number of anchor vectors
            use_mlp: False = V2 (plain concat+linear), True = V2b (MLP on mixture before concat)
            reference_weight: tensor to match init stats from (e.g. embedding.weight.data)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.use_mlp = use_mlp

        self.hub_embeddings = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.log_logit_scale = nn.Parameter(torch.tensor(math.log(14.0)))

        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.GELU(),
            )

        self.linear_out = nn.Linear(2 * embedding_dim, embedding_dim)

        self._init_weights(reference_weight)

    def _init_weights(self, reference_weight: torch.Tensor = None) -> None:
        if reference_weight is not None:
            std = reference_weight.std().item()
            mean = reference_weight.mean().item()
            self.hub_embeddings.data.normal_(mean=mean, std=std)
        else:
            nn.init.xavier_uniform_(self.hub_embeddings.data)

        d = self.embedding_dim
        with torch.no_grad():
            self.linear_out.weight[:, :d].copy_(torch.eye(d))
            self.linear_out.weight[:, d:].zero_()
            self.linear_out.bias.zero_()

        if self.use_mlp:
            nn.init.xavier_uniform_(self.mlp[0].weight)
            nn.init.zeros_(self.mlp[0].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = F.normalize(x, dim=-1)
        k = F.normalize(self.hub_embeddings, dim=-1)
        scale = self.log_logit_scale.exp().clamp(max=100.0)
        logits = (q @ k.T) * scale
        weights = logits.softmax(dim=-1)
        mixture = weights @ self.hub_embeddings

        if self.use_mlp:
            mixture = self.mlp(mixture)

        return self.linear_out(torch.cat([x, mixture], dim=-1))

    def compute_diagnostics(self, x: torch.Tensor) -> dict:
        with torch.no_grad():
            x_f = x.float()
            q = F.normalize(x_f, dim=-1)
            k = F.normalize(self.hub_embeddings.float(), dim=-1)
            scale = self.log_logit_scale.float().exp().clamp(max=100.0)
            logits = (q @ k.T) * scale
            weights = logits.softmax(dim=-1)
            mixture = weights @ self.hub_embeddings.float()

            if self.use_mlp:
                mixture_transformed = F.gelu(
                    F.linear(mixture, self.mlp[0].weight.float(), self.mlp[0].bias.float())
                )
            else:
                mixture_transformed = mixture

            linear_out_result = F.linear(
                torch.cat([x_f, mixture_transformed], dim=-1),
                self.linear_out.weight.float(),
                self.linear_out.bias.float(),
            )
            contribution = linear_out_result - x_f

            entropy = -(weights * weights.clamp(min=1e-12).log()).sum(dim=-1)
            uniform_entropy = math.log(self.num_embeddings)

            anchor_mass = weights.mean(dim=tuple(range(weights.dim() - 1)))
            top10_mass = anchor_mass.topk(min(10, self.num_embeddings)).values.sum().item()
            uniform_mass = 1.0 / self.num_embeddings
            dead_fraction = (anchor_mass < 0.1 * uniform_mass).float().mean().item()

            contrib_norm = contribution.norm(dim=-1)
            token_norm = x_f.norm(dim=-1)
            norm_ratio = (contrib_norm / token_norm.clamp(min=1e-8)).mean().item()

            d = self.embedding_dim
            w_mix_norm = self.linear_out.weight[:, d:].float().norm().item()

            n_sample = min(100, self.num_embeddings)
            sampled = F.normalize(self.hub_embeddings[:n_sample].float(), dim=-1)
            pairwise_cos = (sampled @ sampled.T).triu(diagonal=1)
            mask = torch.triu(torch.ones_like(pairwise_cos, dtype=torch.bool), diagonal=1)
            mean_pairwise_cos = pairwise_cos[mask].mean().item()

            return {
                "logit_std": logits.std().item(),
                "entropy_mean": entropy.mean().item(),
                "entropy_std": entropy.std().item(),
                "uniform_entropy": uniform_entropy,
                "effective_anchors": math.exp(entropy.mean().item()),
                "max_weight_mean": weights.max(dim=-1).values.mean().item(),
                "logit_scale": scale.item(),
                "top10_anchor_mass": top10_mass,
                "dead_anchor_fraction": dead_fraction,
                "norm_ratio": norm_ratio,
                "w_mix_norm": w_mix_norm,
                "anchor_pairwise_cosine": mean_pairwise_cos,
            }
