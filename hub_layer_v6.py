"""EmbHub V6 — stochastic replacement with curriculum (V6 and V6f).

V6:  Stochastic per-token replacement — 50/40/10 (plain/both/anchors-only).
V6f: V6 + small codebook (N=64-128) + norm-capped residual.

Both use renormalized top-k selection over decoupled keys/values.
NO safe init — curriculum replaces it (anneal from 100/0/0 to target mix).
At inference: always uses combined form (deterministic).

Training modes (per token, per step):
  ~50%: plain token embedding (baseline path)
  ~40%: combined (tok_emb + concept for V6; concept + capped_resid for V6f)
  ~10%: concept ONLY (anchors must carry real content — the teeth)

Curriculum: start at 100/0/0, anneal to target over anneal_steps.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbHubV6(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int = 128,
        top_k: int = 10,
        use_residual_cap: bool = False,
        r_budget: float = 0.3,
        p_only: float = 0.10,
        p_both: float = 0.40,
        anneal_steps: int = 2000,
        reference_weight: torch.Tensor = None,
    ):
        """
        Args:
            embedding_dim: dimension of input vectors
            num_embeddings: N anchors (1000 for V6, 64-128 for V6f)
            top_k: number of top-k anchors for retrieval
            use_residual_cap: False = V6 (plain add), True = V6f (norm-capped residual)
            r_budget: max residual norm as fraction of concept norm (V6f only)
            p_only: target probability for concept-only mode
            p_both: target probability for combined mode
            anneal_steps: steps to ramp from 100/0/0 to target mix
            reference_weight: tensor to match init stats from (e.g. embedding.weight.data)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.top_k = min(top_k, num_embeddings)
        self.use_residual_cap = use_residual_cap
        self.r_budget = r_budget
        self.p_only = p_only
        self.p_both = p_both
        self.anneal_steps = anneal_steps

        self.anchor_keys = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.anchor_values = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.log_logit_scale = nn.Parameter(torch.tensor(math.log(14.0)))

        self.register_buffer("current_step", torch.tensor(0, dtype=torch.long))

        self._init_weights(reference_weight)

    def _init_weights(self, reference_weight: torch.Tensor = None) -> None:
        if reference_weight is not None:
            std = reference_weight.std().item()
            mean = reference_weight.mean().item()
            self.anchor_keys.data.normal_(mean=mean, std=std)
            self.anchor_values.data.normal_(mean=mean, std=std)
        else:
            nn.init.xavier_uniform_(self.anchor_keys.data)
            nn.init.xavier_uniform_(self.anchor_values.data)

    def _retrieve_concept(self, x: torch.Tensor):
        """Renormalized top-k retrieval. Returns concept (*, d), weights, indices."""
        q = F.normalize(x, dim=-1)
        k = F.normalize(self.anchor_keys, dim=-1)
        scale = self.log_logit_scale.exp().clamp(max=100.0)
        logits = (q @ k.T) * scale
        weights = logits.softmax(dim=-1)

        topk_weights, topk_indices = weights.topk(self.top_k, dim=-1)
        w_norm = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        anchors = self.anchor_values[topk_indices]
        concept = (w_norm.unsqueeze(-1) * anchors).sum(dim=-2)

        return concept, weights, topk_weights, topk_indices

    def _cap_residual(self, x: torch.Tensor, concept: torch.Tensor):
        """Norm-cap x to at most r_budget * ||concept||. Returns capped residual."""
        max_norm = self.r_budget * concept.norm(dim=-1, keepdim=True)
        x_norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale_dn = (max_norm / x_norm).clamp(max=1.0)
        return x * scale_dn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        concept, weights, topk_weights, topk_indices = self._retrieve_concept(x)

        if self.use_residual_cap:
            resid = self._cap_residual(x, concept)
            combined = concept + resid
        else:
            combined = x + concept

        if not self.training:
            return combined

        ramp = min(1.0, self.current_step.item() / max(1, self.anneal_steps))
        cur_p_only = self.p_only * ramp
        cur_p_both = self.p_both * ramp

        r = torch.rand(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
        out = torch.where(r < cur_p_only, concept,
              torch.where(r < cur_p_only + cur_p_both, combined, x))
        return out

    def compute_diagnostics(self, x: torch.Tensor) -> dict:
        with torch.no_grad():
            x_f = x.float()
            q = F.normalize(x_f, dim=-1)
            k = F.normalize(self.anchor_keys.float(), dim=-1)
            scale = self.log_logit_scale.float().exp().clamp(max=100.0)
            logits = (q @ k.T) * scale
            weights = logits.softmax(dim=-1)

            topk_weights, topk_indices = weights.topk(self.top_k, dim=-1)
            w_norm = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)

            anchors = self.anchor_values.float()[topk_indices]
            concept = (w_norm.unsqueeze(-1) * anchors).sum(dim=-2)

            if self.use_residual_cap:
                resid = self._cap_residual(x_f, concept)
                combined = concept + resid
            else:
                combined = x_f + concept

            contribution = combined - x_f

            entropy = -(weights * weights.clamp(min=1e-12).log()).sum(dim=-1)
            uniform_entropy = math.log(self.num_embeddings)

            anchor_mass = weights.mean(dim=tuple(range(weights.dim() - 1)))
            top10_mass = anchor_mass.topk(min(10, self.num_embeddings)).values.sum().item()
            uniform_mass = 1.0 / self.num_embeddings
            dead_fraction = (anchor_mass < 0.1 * uniform_mass).float().mean().item()

            topk_mass_total = topk_weights.sum(dim=-1).mean().item()

            contrib_norm = contribution.norm(dim=-1)
            token_norm = x_f.norm(dim=-1)
            norm_ratio = (contrib_norm / token_norm.clamp(min=1e-8)).mean().item()

            concept_norm = concept.norm(dim=-1).mean().item()
            ramp = min(1.0, self.current_step.item() / max(1, self.anneal_steps))

            n_sample = min(100, self.num_embeddings)
            sampled = F.normalize(self.anchor_keys[:n_sample].float(), dim=-1)
            pairwise_cos = (sampled @ sampled.T).triu(diagonal=1)
            mask = torch.triu(torch.ones_like(pairwise_cos, dtype=torch.bool), diagonal=1)
            mean_pairwise_cos = pairwise_cos[mask].mean().item()

            result = {
                "logit_std": logits.std().item(),
                "entropy_mean": entropy.mean().item(),
                "entropy_std": entropy.std().item(),
                "uniform_entropy": uniform_entropy,
                "effective_anchors": math.exp(entropy.mean().item()),
                "max_weight_mean": weights.max(dim=-1).values.mean().item(),
                "logit_scale": scale.item(),
                "top10_anchor_mass": top10_mass,
                "dead_anchor_fraction": dead_fraction,
                "topk_mass_total": topk_mass_total,
                "norm_ratio": norm_ratio,
                "concept_norm": concept_norm,
                "curriculum_ramp": ramp,
                "anchor_pairwise_cosine": mean_pairwise_cos,
            }

            if self.use_residual_cap:
                resid_norm = resid.norm(dim=-1).mean().item()
                result["resid_norm"] = resid_norm
                result["resid_concept_ratio"] = resid_norm / max(concept_norm, 1e-8)

            return result
