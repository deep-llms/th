"""EmbHub diagnostics — Tests 1, 2, and 4 from the test plan.

Verifies the cosine + learnable temperature EmbHub implementation before
committing to a full training run.

Usage:
  python diagnostics/embhub_diagnostics.py
  python diagnostics/embhub_diagnostics.py --config-name Qwen/Qwen3-0.6B
  python diagnostics/embhub_diagnostics.py --skip-test4
"""

import argparse
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from hub_layer_v2 import EmbHub
from model_wrapper_v2 import inject_embhub


SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog. This is a common sentence used for testing.",
    "Machine learning models require large amounts of data to train effectively.",
    "Cross-lingual transfer learning enables models to work across multiple languages.",
    "The Eiffel Tower is located in Paris, France and was built in 1889.",
    "Quantum computing promises to solve problems that are intractable for classical computers.",
    "Natural language processing has made significant advances in recent years.",
    "The global economy is influenced by many interconnected factors and policies.",
    "Renewable energy sources include solar, wind, and hydroelectric power generation.",
] * 4


def print_header(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def print_result(name, value, expected=None, pass_range=None):
    status = ""
    if pass_range is not None:
        lo, hi = pass_range
        passed = lo <= value <= hi
        status = " ✓ PASS" if passed else " ✗ FAIL"
    if isinstance(value, float):
        line = f"  {name:<40s} {value:>12.6f}{status}"
    else:
        line = f"  {name:<40s} {value!s:>12s}{status}"
    if expected is not None:
        line += f"  (expected: {expected})"
    print(line)
    return status != " ✗ FAIL"


# -----------------------------------------------------------------------
# Test 1 — Wiring sanity
# -----------------------------------------------------------------------

def test1_wiring_sanity(config_name, num_hub=1000):
    print_header("Test 1 — Wiring Sanity (alpha=0 must match base model)")

    config = AutoConfig.from_pretrained(config_name)

    torch.manual_seed(42)
    model_base = AutoModelForCausalLM.from_config(config).float().eval()

    torch.manual_seed(42)
    model_hub = AutoModelForCausalLM.from_config(config).float().eval()
    inject_embhub(model_hub, num_embeddings=num_hub, alpha=0.0)

    input_ids = torch.randint(0, config.vocab_size, (2, 64))

    with torch.no_grad():
        logits_base = model_base(input_ids).logits
        logits_hub = model_hub(input_ids).logits

    max_diff = (logits_base - logits_hub).abs().max().item()
    mean_diff = (logits_base - logits_hub).abs().mean().item()

    passed = True
    passed &= print_result("Max absolute difference", max_diff, pass_range=(0, 1e-6))
    passed &= print_result("Mean absolute difference", mean_diff)

    del model_base, model_hub
    return passed


# -----------------------------------------------------------------------
# Test 2 — Static initialization check
# -----------------------------------------------------------------------

def test2_static_init(config_name, tokenizer_name, num_hub=1000):
    print_header("Test 2 — Static Initialization Check")

    config = AutoConfig.from_pretrained(config_name)
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_config(config).float().eval()
    hub = inject_embhub(model, num_embeddings=num_hub, alpha=0.05)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    inputs = tokenizer(SAMPLE_TEXTS, return_tensors="pt", padding=True, truncation=True, max_length=256)

    embedding = model.get_input_embeddings()
    with torch.no_grad():
        token_emb = embedding(inputs["input_ids"]).float()
        diag = hub.compute_diagnostics(token_emb)

    print("  --- New formula (cosine + learnable temperature) ---")
    passed = True
    passed &= print_result("Logit scale (exp(log_logit_scale))", diag["logit_scale"], expected="14.0")
    passed &= print_result("Logit std", diag["logit_std"], expected="0.3-0.6", pass_range=(0.3, 0.6))
    passed &= print_result("Entropy mean", diag["entropy_mean"], expected="6.70-6.88", pass_range=(6.70, 6.88))
    passed &= print_result("Entropy std", diag["entropy_std"])
    passed &= print_result("Uniform entropy (log 1000)", diag["uniform_entropy"], expected="6.907")
    passed &= print_result("Effective anchor count", diag["effective_anchors"], expected="~800-970")
    passed &= print_result("Mean max weight", diag["max_weight_mean"], expected="~0.003-0.006", pass_range=(0.002, 0.010))
    passed &= print_result("Norm ratio (||alpha*contrib|| / ||tok||)", diag["norm_ratio"])
    passed &= print_result("Anchor pairwise cosine", diag["anchor_pairwise_cosine"])

    # --- Compare with old formula ---
    print("\n  --- Old formula (dot-product / sqrt(d)) for comparison ---")
    with torch.no_grad():
        old_scale = 1.0 / math.sqrt(hub.embedding_dim)
        old_logits = (token_emb @ hub.hub_embeddings.float().T) * old_scale
        old_weights = old_logits.softmax(dim=-1)
        old_entropy = -(old_weights * old_weights.clamp(min=1e-12).log()).sum(dim=-1)

    print_result("Old logit std", old_logits.std().item(), expected="~0.0004 (uniform)")
    print_result("Old entropy mean", old_entropy.mean().item(), expected="~6.907 (uniform)")
    print_result("Old effective anchors", math.exp(old_entropy.mean().item()), expected="~1000 (all equal)")
    print_result("Old max weight mean", old_weights.max(dim=-1).values.mean().item(), expected="~0.001 (uniform)")

    del model
    return passed


# -----------------------------------------------------------------------
# Test 4 — Gradient symmetry contrast (optional)
# -----------------------------------------------------------------------

def test4_gradient_symmetry(config_name, num_hub=1000):
    print_header("Test 4 — Gradient Symmetry Contrast (old vs new formula)")

    config = AutoConfig.from_pretrained(config_name)
    input_ids = torch.randint(0, config.vocab_size, (2, 64))

    results = {}
    for formula_name, use_old in [("New (cosine+temp)", False), ("Old (dot/sqrt(d))", True)]:
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config).float()
        hub = inject_embhub(model, num_embeddings=num_hub, alpha=0.05)

        if use_old:
            # Monkey-patch forward to use old formula
            old_scale = 1.0 / math.sqrt(hub.embedding_dim)
            def old_forward(self, token_emb, _scale=old_scale):
                weights = torch.softmax(token_emb @ self.hub_embeddings.t() * _scale, dim=-1)
                return token_emb + self.alpha * (weights @ self.hub_embeddings)
            import types
            hub.forward = types.MethodType(old_forward, hub)

        model.train()
        outputs = model(input_ids, labels=input_ids)
        outputs.loss.backward()

        grads = hub.hub_embeddings.grad
        grad_norms = grads.norm(dim=-1)
        grad_normed = F.normalize(grads, dim=-1)

        # Sample pairwise cosines between anchor gradients
        idx = torch.randperm(num_hub)[:50]
        sampled = grad_normed[idx]
        pairwise = (sampled @ sampled.T).triu(diagonal=1)
        mask = torch.triu(torch.ones_like(pairwise, dtype=torch.bool), diagonal=1)
        mean_cos = pairwise[mask].mean().item()

        cv = (grad_norms.std() / grad_norms.mean()).item()

        results[formula_name] = {"mean_pairwise_cosine": mean_cos, "grad_norm_cv": cv}

        del model

    passed = True
    for name, r in results.items():
        print(f"  --- {name} ---")
        if "Old" in name:
            passed &= print_result("Grad pairwise cosine", r["mean_pairwise_cosine"], expected="~1.0 (clones)")
            passed &= print_result("Grad norm CV", r["grad_norm_cv"], expected="~0 (identical)")
        else:
            passed &= print_result("Grad pairwise cosine", r["mean_pairwise_cosine"], expected="<1.0 (diverse)")
            passed &= print_result("Grad norm CV", r["grad_norm_cv"], expected=">0.05 (differentiated)", pass_range=(0.05, 100))

    return passed


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EmbHub diagnostics")
    parser.add_argument("--config-name", default="Qwen/Qwen3-0.6B", help="Model config")
    parser.add_argument("--tokenizer-name", default=None, help="Tokenizer (default: same as config)")
    parser.add_argument("--num-hub", type=int, default=1000, help="Number of hub embeddings")
    parser.add_argument("--skip-test4", action="store_true", help="Skip gradient symmetry test")
    args = parser.parse_args()

    if args.tokenizer_name is None:
        args.tokenizer_name = args.config_name

    all_passed = True

    # Test 1
    t1 = test1_wiring_sanity(args.config_name, args.num_hub)
    all_passed &= t1

    # Test 2
    t2 = test2_static_init(args.config_name, args.tokenizer_name, args.num_hub)
    all_passed &= t2

    # Test 4 (optional)
    t4 = True
    if not args.skip_test4:
        t4 = test4_gradient_symmetry(args.config_name, args.num_hub)
        all_passed &= t4

    # Summary
    print_header("Summary")
    print(f"  Test 1 (Wiring sanity):        {'PASS' if t1 else 'FAIL'}")
    print(f"  Test 2 (Static init):          {'PASS' if t2 else 'FAIL'}")
    if not args.skip_test4:
        print(f"  Test 4 (Gradient symmetry):    {'PASS' if t4 else 'FAIL'}")
    print()

    if all_passed:
        print("  ✓ All tests passed. Proceed to smoke training run (Test 3).")
    else:
        print("  ✗ Some tests failed. Fix before proceeding.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
