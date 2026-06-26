import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbHub(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int = 1000,
        alpha: float = 0.05,
        reference_embedding: nn.Embedding = None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.alpha = alpha
        self.hub_embeddings = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.log_logit_scale = nn.Parameter(torch.tensor(math.log(14.0)))
        self._init_weights(reference_embedding)

    def _init_weights(self, reference_embedding: nn.Embedding = None) -> None:
        if reference_embedding is not None:
            std = reference_embedding.weight.std().item()
            mean = reference_embedding.weight.mean().item()
            self.hub_embeddings.data.normal_(mean=mean, std=std)
        else:
            nn.init.xavier_uniform_(self.hub_embeddings.data)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        if self.alpha == 0.0:
            return token_embeddings
        q = F.normalize(token_embeddings, dim=-1)
        k = F.normalize(self.hub_embeddings, dim=-1)
        scale = self.log_logit_scale.exp().clamp(max=100.0)
        logits = (q @ k.T) * scale
        weights = logits.softmax(dim=-1)
        hub_contribution = weights @ self.hub_embeddings
        return token_embeddings + self.alpha * hub_contribution

    def compute_diagnostics(self, token_embeddings: torch.Tensor) -> dict:
        with torch.no_grad():
            q = F.normalize(token_embeddings.float(), dim=-1)
            k = F.normalize(self.hub_embeddings.float(), dim=-1)
            scale = self.log_logit_scale.float().exp().clamp(max=100.0)
            logits = (q @ k.T) * scale
            weights = logits.softmax(dim=-1)
            contribution = weights @ self.hub_embeddings.float()

            entropy = -(weights * weights.clamp(min=1e-12).log()).sum(dim=-1)
            uniform_entropy = math.log(self.num_embeddings)

            anchor_mass = weights.mean(dim=(0, 1))
            top10_mass = anchor_mass.topk(10).values.sum().item()
            uniform_mass = 1.0 / self.num_embeddings
            dead_fraction = (anchor_mass < 0.1 * uniform_mass).float().mean().item()

            contrib_norm = (self.alpha * contribution).norm(dim=-1)
            token_norm = token_embeddings.float().norm(dim=-1)
            norm_ratio = (contrib_norm / token_norm.clamp(min=1e-8)).mean().item()

            # Fixed deterministic subset (no RNG side-effect on the training CUDA
            # generator; also gives a stable subset for trend tracking over steps).
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
                "anchor_pairwise_cosine": mean_pairwise_cos,
            }
