"""Check anchor geometry and log_logit_scale gradient status.

Loads a trained EmbHub checkpoint and runs the diagnostic checks
suggested by the analysis of the v2 smoke run.

Usage:
  python diagnostics/check_anchor_geometry.py --checkpoint /path/to/ckpt
"""

import argparse
import itertools
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from model_wrapper_v2 import inject_embhub, EMBHUB_WEIGHTS_NAME, EMBHUB_CONFIG_NAME


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    # Load model with embhub
    config = AutoConfig.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, config=config)

    embhub_cfg_path = os.path.join(args.checkpoint, EMBHUB_CONFIG_NAME)
    embhub_wt_path = os.path.join(args.checkpoint, EMBHUB_WEIGHTS_NAME)

    with open(embhub_cfg_path) as f:
        hub_cfg = json.load(f)
    hub = inject_embhub(model, num_embeddings=hub_cfg["num_embeddings"], alpha=hub_cfg["alpha"])
    hub.load_state_dict(torch.load(embhub_wt_path, map_location="cpu", weights_only=True))

    print("=" * 60)
    print("  ANCHOR GEOMETRY CHECK")
    print("=" * 60)

    A = hub.hub_embeddings.detach().float()  # (N, d)
    N, d = A.shape
    print(f"\n  Anchors: {N} x {d}")

    # 1. Shared mean vs residual magnitude
    mu = A.mean(0)
    res = A - mu
    shared_norm = mu.norm().item()
    residual_norms = res.norm(dim=1)
    mean_residual_norm = residual_norms.mean().item()
    ratio = shared_norm / mean_residual_norm

    print(f"\n  Shared mean ||μ||:           {shared_norm:.4f}")
    print(f"  Mean residual ||aⱼ - μ||:    {mean_residual_norm:.4f}")
    print(f"  Ratio (shared / residual):   {ratio:.4f}")
    if ratio > 1:
        print(f"  → Shared component DOMINATES (ratio > 1)")
    else:
        print(f"  → Residuals dominate (ratio < 1)")

    # 2. PCA of residuals
    U, S, V = torch.pca_lowrank(res, q=20)
    total_var = (res ** 2).sum().item()
    cumvar = (S ** 2).cumsum(0) / total_var
    print(f"\n  PCA of residuals (top 10 singular values):")
    print(f"  {'PC':>4}  {'Singular Val':>12}  {'Cumulative Var %':>16}")
    for i in range(min(10, len(S))):
        print(f"  {i+1:>4}  {S[i].item():>12.4f}  {cumvar[i].item() * 100:>15.2f}%")
    if S[0] / S[1] > 5:
        print(f"  → Near rank-1 residuals (PC1/PC2 = {S[0]/S[1]:.1f}x) — diversity is low")
    else:
        print(f"  → Residuals are multi-dimensional (PC1/PC2 = {S[0]/S[1]:.1f}x) — diversity intact")

    # 3. Raw pairwise cosine (confirms the 0.51 from diagnostics)
    An = F.normalize(A, dim=-1)
    random.seed(42)
    ps = random.sample(range(N), 200)
    raw_cos = torch.stack([An[i] @ An[j] for i, j in itertools.combinations(ps, 2)])
    print(f"\n  Raw pairwise cosine:         {raw_cos.mean().item():.4f} (±{raw_cos.std().item():.4f})")

    # 4. Residual-only pairwise cosine
    Rn = F.normalize(res, dim=-1)
    res_cos = torch.stack([Rn[i] @ Rn[j] for i, j in itertools.combinations(ps, 2)])
    print(f"  Residual pairwise cosine:    {res_cos.mean().item():.4f} (±{res_cos.std().item():.4f})")

    if abs(res_cos.mean().item()) < 0.1 and raw_cos.mean().item() > 0.3:
        print(f"  → CONFIRMED: anchors are distinct directions on a shared offset")
        print(f"    (raw cosine high due to shared μ, residual cosine near 0 = diverse)")
    else:
        print(f"  → Anchors may be genuinely collapsing")

    # 5. log_logit_scale status
    print(f"\n" + "=" * 60)
    print(f"  LOG_LOGIT_SCALE CHECK")
    print(f"=" * 60)

    scale_param = hub.log_logit_scale
    print(f"\n  Value:          {scale_param.item():.6f}")
    print(f"  exp(value):     {scale_param.exp().item():.6f}")
    print(f"  requires_grad:  {scale_param.requires_grad}")

    # Check if it's in the model's named_parameters
    found = False
    for name, p in model.named_parameters():
        if p is scale_param:
            found = True
            print(f"  In named_params: Yes ('{name}')")
            break
    if not found:
        print(f"  In named_params: NO — this is the bug!")

    # Quick gradient check: one forward-backward
    model.train()
    dummy_ids = torch.randint(0, config.vocab_size, (1, 32))
    outputs = model(dummy_ids, labels=dummy_ids)
    outputs.loss.backward()

    if scale_param.grad is not None:
        print(f"  Gradient:       {scale_param.grad.item():.8f}")
        if abs(scale_param.grad.item()) < 1e-8:
            print(f"  → Gradient is effectively ZERO (vanishing gradient problem)")
        else:
            print(f"  → Gradient is non-zero (param should be learning)")
    else:
        print(f"  Gradient:       None — NOT RECEIVING GRADIENTS!")
        print(f"  → This is the bug: param is not connected to the computation graph")

    print()


if __name__ == "__main__":
    main()
