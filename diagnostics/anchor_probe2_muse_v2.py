"""Probe 2 v2 — Improved cross-lingual analysis per reviewer recommendations.

Changes from v1 (anchor_probe2_muse.py):
1. Mean-pooling for multi-token words (already in v1, confirmed here)
2. JS-divergence restricted to alive-anchor support as PRIMARY metric;
   random control drawn from same alive pool to remove dead-set confound.
   Top-k Jaccard demoted to secondary.
3. NEW: Post-hub embedding cosine similarity test — does the hub pull
   translation equivalents closer in representation space? Compares
   with hub (alpha) vs without hub (alpha=0).

Usage:
  python diagnostics/anchor_probe2_muse_v2.py \
      --checkpoints /path/to/checkpoint-1500 /path/to/checkpoint-3250 \
      --translations temp/frequent_translations.json \
      --output temp/probe2_muse_v2_RESULTS.md
"""

import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from scipy import stats
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from model_wrapper_v2 import inject_embhub, EMBHUB_WEIGHTS_NAME, EMBHUB_CONFIG_NAME

LANGS = ["en", "vi", "zh", "ru", "de", "ar"]


def load_checkpoint(checkpoint_path, device="cpu"):
    config = AutoConfig.from_pretrained(checkpoint_path)
    embhub_cfg_path = os.path.join(checkpoint_path, EMBHUB_CONFIG_NAME)
    embhub_wt_path = os.path.join(checkpoint_path, EMBHUB_WEIGHTS_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, config=config, torch_dtype=torch.bfloat16
    )

    with open(embhub_cfg_path) as f:
        hub_cfg = json.load(f)
    hub = inject_embhub(model, num_embeddings=hub_cfg["num_embeddings"], alpha=hub_cfg["alpha"])
    hub.load_state_dict(torch.load(embhub_wt_path, map_location="cpu", weights_only=True))

    model.to(device)
    model.eval()
    return model, hub, hub_cfg["alpha"]


def get_anchor_weights(hub, token_embeddings):
    with torch.no_grad():
        q = F.normalize(token_embeddings.float(), dim=-1)
        k = F.normalize(hub.hub_embeddings.float(), dim=-1)
        scale = hub.log_logit_scale.float().exp().clamp(max=100.0)
        logits = (q @ k.T) * scale
        weights = logits.softmax(dim=-1)
    return weights


def get_word_data(hub, embedding, tokenizer, word, device, alpha):
    """Get anchor distribution AND post-hub embedding for a word."""
    ids = tokenizer(word, add_special_tokens=False, return_tensors="pt")["input_ids"]
    ids = ids.to(device)
    with torch.no_grad():
        token_emb = embedding(ids)  # (1, n_tokens, dim)
        weights = get_anchor_weights(hub, token_emb)
        anchor_dist = weights.squeeze(0).mean(dim=0)  # (num_anchors,)

        # Post-hub embedding: token_emb + alpha * (weights @ anchors)
        hub_contribution = weights @ hub.hub_embeddings.float()
        emb_with_hub = token_emb.float() + alpha * hub_contribution
        emb_with_hub = emb_with_hub.squeeze(0).mean(dim=0)  # (dim,)

        # Without hub (alpha=0): just the raw token embedding
        emb_without_hub = token_emb.float().squeeze(0).mean(dim=0)  # (dim,)

    return {
        "anchor_dist": anchor_dist,
        "emb_with_hub": emb_with_hub,
        "emb_without_hub": emb_without_hub,
        "n_tokens": ids.shape[1],
    }


# ---------------------------------------------------------------------------
# Alive-anchor support utilities
# ---------------------------------------------------------------------------

def compute_alive_mask(all_dists, threshold_frac=0.1):
    """Compute which anchors are alive (receive meaningful mass across all words).

    An anchor is alive if its average mass across all word distributions
    exceeds threshold_frac * uniform_mass.
    """
    if not all_dists:
        return None
    stacked = torch.stack(all_dists)  # (n_words, num_anchors)
    avg_mass = stacked.mean(dim=0)
    uniform_mass = 1.0 / stacked.shape[1]
    alive = avg_mass >= (threshold_frac * uniform_mass)
    return alive


def restrict_to_alive(dist, alive_mask):
    """Zero out dead anchors and renormalize."""
    restricted = dist * alive_mask.float()
    total = restricted.sum()
    if total > 0:
        restricted = restricted / total
    return restricted


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def js_divergence(p, q):
    m = 0.5 * (p + q)
    eps = 1e-12
    kl_pm = (p * (p.clamp(min=eps) / m.clamp(min=eps)).log()).sum()
    kl_qm = (q * (q.clamp(min=eps) / m.clamp(min=eps)).log()).sum()
    return (0.5 * (kl_pm + kl_qm)).item()


def topk_jaccard(w1, w2, k=10):
    s1 = set(w1.topk(k).indices.tolist())
    s2 = set(w2.topk(k).indices.tolist())
    return len(s1 & s2) / len(s1 | s2) if len(s1 | s2) > 0 else 0


def cosine_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


# ---------------------------------------------------------------------------
# Tuple loading
# ---------------------------------------------------------------------------

def build_translation_tuples(trans_data):
    tuples = []
    all_five = trans_data.get("all_five_tuples", {})
    for en_word, lang_words in all_five.items():
        entry = {"en": en_word}
        entry.update(lang_words)
        tuples.append(entry)

    if tuples:
        print(f"  All-5-lang tuples: {len(tuples)}")
        return tuples

    multi = trans_data.get("multi_lang_tuples", {})
    for en_word, lang_words in multi.items():
        entry = {"en": en_word}
        entry.update(lang_words)
        tuples.append(entry)

    print(f"  Multi-lang tuples (>=2): {len(tuples)}")
    return tuples


# ---------------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------------

def probe2_v2(hub, embedding, tokenizer, device, tuples, alpha, k=10):
    random.seed(42)

    # Gather all word data
    word_data = {}  # (tuple_idx, lang) -> {anchor_dist, emb_with_hub, emb_without_hub, n_tokens}

    print("  Computing word embeddings and anchor distributions...")
    for t_idx, tup in enumerate(tuples):
        for lang in LANGS:
            if lang not in tup:
                continue
            word = tup[lang]
            word_data[(t_idx, lang)] = get_word_data(hub, embedding, tokenizer, word, device, alpha)

    # Token count stats
    single_per_lang = {lang: 0 for lang in LANGS}
    total_per_lang = {lang: 0 for lang in LANGS}
    for (t_idx, lang), wd in word_data.items():
        total_per_lang[lang] += 1
        if wd["n_tokens"] == 1:
            single_per_lang[lang] += 1
    print(f"  Words per language: {total_per_lang}")
    print(f"  Single-token per language: {single_per_lang}")

    # Compute alive anchor mask from all distributions
    all_dists = [wd["anchor_dist"] for wd in word_data.values()]
    alive_mask = compute_alive_mask(all_dists)
    n_alive = alive_mask.sum().item() if alive_mask is not None else 0
    n_total = alive_mask.shape[0] if alive_mask is not None else 0
    print(f"  Alive anchors: {n_alive}/{n_total} ({n_alive/n_total*100:.1f}%)")

    # Precompute alive-restricted distributions
    word_dists_alive = {}
    for key, wd in word_data.items():
        word_dists_alive[key] = restrict_to_alive(wd["anchor_dist"], alive_mask)

    # ===================================================================
    # TEST A: Anchor distribution overlap (alive-restricted JS as primary)
    # ===================================================================
    print("  Test A: Anchor distribution overlap...")

    trans_js_alive = []
    trans_jaccards = []
    pair_data = {}

    n_tuples = len(tuples)
    for t_idx, tup in enumerate(tuples):
        langs_in_tup = [l for l in LANGS if l in tup and (t_idx, l) in word_data]
        for i, l1 in enumerate(langs_in_tup):
            for l2 in langs_in_tup[i + 1:]:
                d1_alive = word_dists_alive[(t_idx, l1)]
                d2_alive = word_dists_alive[(t_idx, l2)]
                d1_full = word_data[(t_idx, l1)]["anchor_dist"]
                d2_full = word_data[(t_idx, l2)]["anchor_dist"]

                js_a = 1.0 - js_divergence(d1_alive, d2_alive)
                j = topk_jaccard(d1_full, d2_full, k)

                trans_js_alive.append(js_a)
                trans_jaccards.append(j)

                pair_key = f"{min(l1,l2)}-{max(l1,l2)}"
                if pair_key not in pair_data:
                    pair_data[pair_key] = {"js_alive": [], "jaccard": []}
                pair_data[pair_key]["js_alive"].append(js_a)
                pair_data[pair_key]["jaccard"].append(j)

    # Random control: different tuples, same language pairs, same alive pool
    rand_js_alive = []
    rand_jaccards = []
    for _ in range(len(trans_js_alive)):
        t1, t2 = random.sample(range(n_tuples), 2)
        l1, l2 = random.sample(LANGS, 2)
        d1 = word_dists_alive.get((t1, l1))
        d2 = word_dists_alive.get((t2, l2))
        if d1 is not None and d2 is not None:
            rand_js_alive.append(1.0 - js_divergence(d1, d2))
            d1f = word_data[(t1, l1)]["anchor_dist"]
            d2f = word_data[(t2, l2)]["anchor_dist"]
            rand_jaccards.append(topk_jaccard(d1f, d2f, k))

    # Per-pair means
    per_pair_a = {}
    for pair_key, data in sorted(pair_data.items()):
        per_pair_a[pair_key] = {
            "js_alive_mean": sum(data["js_alive"]) / len(data["js_alive"]),
            "jaccard_mean": sum(data["jaccard"]) / len(data["jaccard"]),
            "n_pairs": len(data["js_alive"]),
        }

    # Significance
    pval_js_a = pval_j = 1.0
    if trans_js_alive and rand_js_alive:
        _, pval_js_a = stats.mannwhitneyu(trans_js_alive, rand_js_alive, alternative="greater")
    if trans_jaccards and rand_jaccards:
        _, pval_j = stats.mannwhitneyu(trans_jaccards, rand_jaccards, alternative="greater")

    mean_t_js = sum(trans_js_alive) / len(trans_js_alive) if trans_js_alive else 0
    mean_r_js = sum(rand_js_alive) / len(rand_js_alive) if rand_js_alive else 0
    mean_t_j = sum(trans_jaccards) / len(trans_jaccards) if trans_jaccards else 0
    mean_r_j = sum(rand_jaccards) / len(rand_jaccards) if rand_jaccards else 0

    test_a = {
        "n_alive_anchors": int(n_alive),
        "n_total_anchors": int(n_total),
        "translation_js_alive_mean": mean_t_js,
        "random_js_alive_mean": mean_r_js,
        "js_alive_gap": mean_t_js - mean_r_js,
        "js_alive_pvalue": pval_js_a,
        "translation_jaccard_mean": mean_t_j,
        "random_jaccard_mean": mean_r_j,
        "jaccard_gap": mean_t_j - mean_r_j,
        "jaccard_pvalue": pval_j,
        "per_pair": per_pair_a,
        "n_translation_comparisons": len(trans_js_alive),
        "n_random_comparisons": len(rand_js_alive),
    }

    # ===================================================================
    # TEST B: Post-hub embedding cosine (the key question)
    # ===================================================================
    print("  Test B: Post-hub embedding cosine similarity...")

    trans_cos_with = []
    trans_cos_without = []
    pair_data_b = {}

    for t_idx, tup in enumerate(tuples):
        langs_in_tup = [l for l in LANGS if l in tup and (t_idx, l) in word_data]
        for i, l1 in enumerate(langs_in_tup):
            for l2 in langs_in_tup[i + 1:]:
                wd1 = word_data[(t_idx, l1)]
                wd2 = word_data[(t_idx, l2)]

                cos_w = cosine_sim(wd1["emb_with_hub"], wd2["emb_with_hub"])
                cos_wo = cosine_sim(wd1["emb_without_hub"], wd2["emb_without_hub"])

                trans_cos_with.append(cos_w)
                trans_cos_without.append(cos_wo)

                pair_key = f"{min(l1,l2)}-{max(l1,l2)}"
                if pair_key not in pair_data_b:
                    pair_data_b[pair_key] = {"cos_with": [], "cos_without": []}
                pair_data_b[pair_key]["cos_with"].append(cos_w)
                pair_data_b[pair_key]["cos_without"].append(cos_wo)

    # Random control for embedding cosine
    rand_cos_with = []
    rand_cos_without = []
    for _ in range(len(trans_cos_with)):
        t1, t2 = random.sample(range(n_tuples), 2)
        l1, l2 = random.sample(LANGS, 2)
        wd1 = word_data.get((t1, l1))
        wd2 = word_data.get((t2, l2))
        if wd1 is not None and wd2 is not None:
            rand_cos_with.append(cosine_sim(wd1["emb_with_hub"], wd2["emb_with_hub"]))
            rand_cos_without.append(cosine_sim(wd1["emb_without_hub"], wd2["emb_without_hub"]))

    # Per-pair means
    per_pair_b = {}
    for pair_key, data in sorted(pair_data_b.items()):
        per_pair_b[pair_key] = {
            "cos_with_mean": sum(data["cos_with"]) / len(data["cos_with"]),
            "cos_without_mean": sum(data["cos_without"]) / len(data["cos_without"]),
            "hub_delta": (sum(data["cos_with"]) - sum(data["cos_without"])) / len(data["cos_with"]),
            "n_pairs": len(data["cos_with"]),
        }

    # Significance: does hub increase cosine for translation pairs?
    pval_hub_helps = 1.0
    if trans_cos_with and trans_cos_without:
        _, pval_hub_helps = stats.wilcoxon(
            trans_cos_with, trans_cos_without, alternative="greater"
        )

    # Significance: translation > random (with hub)?
    pval_trans_vs_rand = 1.0
    if trans_cos_with and rand_cos_with:
        _, pval_trans_vs_rand = stats.mannwhitneyu(
            trans_cos_with, rand_cos_with, alternative="greater"
        )

    # Significance: translation > random (without hub)?
    pval_trans_vs_rand_nohub = 1.0
    if trans_cos_without and rand_cos_without:
        _, pval_trans_vs_rand_nohub = stats.mannwhitneyu(
            trans_cos_without, rand_cos_without, alternative="greater"
        )

    def safe_mean(lst):
        return sum(lst) / len(lst) if lst else 0

    test_b = {
        "trans_cos_with_hub_mean": safe_mean(trans_cos_with),
        "trans_cos_without_hub_mean": safe_mean(trans_cos_without),
        "hub_delta_trans": safe_mean(trans_cos_with) - safe_mean(trans_cos_without),
        "pval_hub_helps_trans": pval_hub_helps,
        "rand_cos_with_hub_mean": safe_mean(rand_cos_with),
        "rand_cos_without_hub_mean": safe_mean(rand_cos_without),
        "hub_delta_rand": safe_mean(rand_cos_with) - safe_mean(rand_cos_without),
        "trans_vs_rand_with_hub_gap": safe_mean(trans_cos_with) - safe_mean(rand_cos_with),
        "pval_trans_vs_rand_with_hub": pval_trans_vs_rand,
        "trans_vs_rand_without_hub_gap": safe_mean(trans_cos_without) - safe_mean(rand_cos_without),
        "pval_trans_vs_rand_without_hub": pval_trans_vs_rand_nohub,
        "per_pair": per_pair_b,
        "n_comparisons": len(trans_cos_with),
    }

    # ===================================================================
    # Examples
    # ===================================================================
    examples = []
    for t_idx in range(min(3, n_tuples)):
        tup = tuples[t_idx]
        ex = {"en": tup.get("en", "?")}
        for lang in LANGS:
            if lang in tup and (t_idx, lang) in word_data:
                wd = word_data[(t_idx, lang)]
                ex[lang] = {
                    "word": tup[lang],
                    "n_tokens": wd["n_tokens"],
                    "top_anchors": wd["anchor_dist"].topk(k).indices.tolist(),
                }
        examples.append(ex)

    return {
        "n_tuples": n_tuples,
        "alpha": alpha,
        "single_token_per_lang": single_per_lang,
        "total_per_lang": total_per_lang,
        "test_a_anchor_overlap": test_a,
        "test_b_embedding_cosine": test_b,
        "examples": examples,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_results(all_results, output_path, n_tuples):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# Probe 2 v2 — MUSE-based Cross-Lingual Analysis (Improved)\n\n")
        f.write(f"Translation tuples: {n_tuples} (from MUSE dictionaries, frequency-filtered)\n\n")
        f.write("Changes from v1:\n")
        f.write("- **Primary metric**: JS-divergence restricted to alive anchors (removes dead-set confound)\n")
        f.write("- **Random control**: drawn from same alive pool\n")
        f.write("- **NEW Test B**: Post-hub embedding cosine — does hub pull translations closer?\n")
        f.write("- Top-k Jaccard demoted to secondary\n\n")

        for step, res in all_results.items():
            ta = res["test_a_anchor_overlap"]
            tb = res["test_b_embedding_cosine"]

            f.write(f"## Checkpoint step {step}\n\n")
            f.write(f"- Alpha: {res['alpha']}\n")
            f.write(f"- Tuples: {res['n_tuples']}\n")
            f.write(f"- Alive anchors: {ta['n_alive_anchors']}/{ta['n_total_anchors']}\n")
            f.write(f"- Words per language: {res['total_per_lang']}\n")
            f.write(f"- Single-token per language: {res['single_token_per_lang']}\n\n")

            # Test A
            f.write("### Test A — Anchor Distribution Overlap\n\n")
            f.write(f"Comparisons: {ta['n_translation_comparisons']} translation, {ta['n_random_comparisons']} random\n\n")
            f.write("| Metric | Translation | Random | Gap | p-value |\n")
            f.write("|--------|------------|--------|-----|--------|\n")
            f.write(f"| **JS sim (alive only)** | {ta['translation_js_alive_mean']:.4f} | {ta['random_js_alive_mean']:.4f} | {ta['js_alive_gap']:+.4f} | {ta['js_alive_pvalue']:.2e} |\n")
            f.write(f"| Top-10 Jaccard (secondary) | {ta['translation_jaccard_mean']:.4f} | {ta['random_jaccard_mean']:.4f} | {ta['jaccard_gap']:+.4f} | {ta['jaccard_pvalue']:.2e} |\n")

            if ta['per_pair']:
                f.write(f"\nPer language-pair:\n\n")
                f.write("| Pair | N | JS alive | Jaccard |\n")
                f.write("|------|---|----------|--------|\n")
                for pair, vals in sorted(ta['per_pair'].items()):
                    f.write(f"| {pair} | {vals['n_pairs']} | {vals['js_alive_mean']:.4f} | {vals['jaccard_mean']:.4f} |\n")

            # Test B
            f.write(f"\n### Test B — Post-Hub Embedding Cosine Similarity\n\n")
            f.write("Does the hub pull translation equivalents closer in representation space?\n\n")
            f.write("| | With Hub | Without Hub | Delta | p-value |\n")
            f.write("|--|---------|------------|-------|--------|\n")
            f.write(f"| Translation pairs | {tb['trans_cos_with_hub_mean']:.4f} | {tb['trans_cos_without_hub_mean']:.4f} | {tb['hub_delta_trans']:+.4f} | {tb['pval_hub_helps_trans']:.2e} |\n")
            f.write(f"| Random pairs | {tb['rand_cos_with_hub_mean']:.4f} | {tb['rand_cos_without_hub_mean']:.4f} | {tb['hub_delta_rand']:+.4f} | — |\n")
            f.write(f"\n| | Translation | Random | Gap | p-value |\n")
            f.write(f"|--|-----------|--------|-----|--------|\n")
            f.write(f"| With hub | {tb['trans_cos_with_hub_mean']:.4f} | {tb['rand_cos_with_hub_mean']:.4f} | {tb['trans_vs_rand_with_hub_gap']:+.4f} | {tb['pval_trans_vs_rand_with_hub']:.2e} |\n")
            f.write(f"| Without hub | {tb['trans_cos_without_hub_mean']:.4f} | {tb['rand_cos_without_hub_mean']:.4f} | {tb['trans_vs_rand_without_hub_gap']:+.4f} | {tb['pval_trans_vs_rand_without_hub']:.2e} |\n")

            if tb['per_pair']:
                f.write(f"\nPer language-pair embedding cosine:\n\n")
                f.write("| Pair | N | With Hub | Without Hub | Hub Delta |\n")
                f.write("|------|---|---------|------------|----------|\n")
                for pair, vals in sorted(tb['per_pair'].items()):
                    f.write(f"| {pair} | {vals['n_pairs']} | {vals['cos_with_mean']:.4f} | {vals['cos_without_mean']:.4f} | {vals['hub_delta']:+.4f} |\n")

            # Examples
            if res.get('examples'):
                f.write(f"\nExamples (top-10 anchors):\n\n")
                for ex in res['examples']:
                    en = ex.get("en", "?")
                    f.write(f"\n**'{en}'**:\n\n")
                    f.write("| Lang | Word | Tokens | Top-10 Anchors |\n")
                    f.write("|------|------|--------|---------------|\n")
                    for lang in LANGS:
                        if lang in ex and isinstance(ex[lang], dict):
                            info = ex[lang]
                            f.write(f"| {lang} | {info['word']} | {info['n_tokens']} | {info['top_anchors']} |\n")

            f.write("\n---\n\n")

    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Probe 2 v2 — improved cross-lingual analysis")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--translations", default="temp/frequent_translations.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="temp/probe2_muse_v2_RESULTS.md")
    args = parser.parse_args()

    with open(args.translations) as f:
        trans_data = json.load(f)
    tuples = build_translation_tuples(trans_data)

    if not tuples:
        print("ERROR: No translation tuples found. Run build_frequent_translations.py first.")
        sys.exit(1)

    print(f"Loaded {len(tuples)} translation tuples")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoints[0])
    all_results = {}

    for ckpt in args.checkpoints:
        step = os.path.basename(ckpt).replace("checkpoint-", "")
        print(f"\n{'='*60}")
        print(f"  Checkpoint: {ckpt} (step {step})")
        print(f"{'='*60}")

        model, hub, alpha = load_checkpoint(ckpt, device=args.device)
        embedding = model.get_input_embeddings()

        res = probe2_v2(hub, embedding, tokenizer, args.device, tuples, alpha)
        all_results[step] = res

        del model, hub
        torch.cuda.empty_cache()

    write_results(all_results, args.output, len(tuples))

    json_path = args.output.replace(".md", ".json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Raw data saved to {json_path}")


if __name__ == "__main__":
    main()
