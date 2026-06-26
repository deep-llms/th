"""Probe 2 v2 — Improved cross-lingual analysis, loanword-filtered version.

Filters out tuples where any non-English translation is identical to the
English word (e.g. love→love in vi/de).

Test A: JS-divergence on alive anchors (primary), top-k Jaccard (secondary)
Test B (corrected): EmbHub vs Baseline token embedding cosine — did training
    through the hub produce embeddings where translations are closer?
Test C: Contribution vector cosine — are hub contributions more aligned for
    translation pairs than random pairs?

Usage:
  python diagnostics/anchor_probe2_muse_v2_no_loan_word.py \
      --checkpoints /opt/dlami/nvme/smoke_tests/S3/checkpoint-1500 ... \
      --baseline-dir /opt/dlami/nvme/checkpoints/qwen3-0.6b-scratch-baseline \
      --translations temp/frequent_translations.json \
      --output temp/probe2_muse_v2_no_loanword_RESULTS.md
"""

import argparse
import json
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


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_embhub_checkpoint(checkpoint_path, device="cpu"):
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


def load_baseline_checkpoint(checkpoint_path, device="cpu"):
    config = AutoConfig.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, config=config, torch_dtype=torch.bfloat16
    )
    model.to(device)
    model.eval()
    return model


def find_baseline_checkpoint(baseline_dir, target_step):
    """Find the closest baseline checkpoint to the target step."""
    best_path = None
    best_diff = float("inf")
    if not os.path.isdir(baseline_dir):
        return None
    for name in os.listdir(baseline_dir):
        if name.startswith("checkpoint-"):
            try:
                step = int(name.split("-")[1])
                diff = abs(step - target_step)
                if diff < best_diff:
                    best_diff = diff
                    best_path = os.path.join(baseline_dir, name)
            except (ValueError, IndexError):
                continue
    return best_path


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------

def get_anchor_weights(hub, token_embeddings):
    with torch.no_grad():
        q = F.normalize(token_embeddings.float(), dim=-1)
        k = F.normalize(hub.hub_embeddings.float(), dim=-1)
        scale = hub.log_logit_scale.float().exp().clamp(max=100.0)
        logits = (q @ k.T) * scale
        weights = logits.softmax(dim=-1)
    return weights


def get_word_embhub_data(hub, embedding, tokenizer, word, device):
    """Get anchor dist, raw token emb, and contribution vector for a word."""
    ids = tokenizer(word, add_special_tokens=False, return_tensors="pt")["input_ids"]
    ids = ids.to(device)
    with torch.no_grad():
        # Use F.embedding to bypass the forward hook (which adds hub contribution)
        token_emb = F.embedding(ids, embedding.weight)
        weights = get_anchor_weights(hub, token_emb)
        anchor_dist = weights.squeeze(0).mean(dim=0)

        contribution = (weights @ hub.hub_embeddings.float()).squeeze(0).mean(dim=0)
        raw_emb = token_emb.float().squeeze(0).mean(dim=0)

    return {
        "anchor_dist": anchor_dist,
        "raw_emb": raw_emb,           # token embedding (no hub)
        "contribution": contribution,   # hub contribution vector (before alpha scaling)
        "n_tokens": ids.shape[1],
    }


def get_word_baseline_emb(embedding, tokenizer, word, device):
    """Get raw token embedding from the baseline model."""
    ids = tokenizer(word, add_special_tokens=False, return_tensors="pt")["input_ids"]
    ids = ids.to(device)
    with torch.no_grad():
        token_emb = embedding(ids)
        raw_emb = token_emb.float().squeeze(0).mean(dim=0)
    return raw_emb


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
# Alive-anchor support
# ---------------------------------------------------------------------------

def compute_alive_mask(all_dists, threshold_frac=0.1):
    if not all_dists:
        return None
    stacked = torch.stack(all_dists)
    avg_mass = stacked.mean(dim=0)
    uniform_mass = 1.0 / stacked.shape[1]
    return avg_mass >= (threshold_frac * uniform_mass)


def restrict_to_alive(dist, alive_mask):
    restricted = dist * alive_mask.float()
    total = restricted.sum()
    if total > 0:
        restricted = restricted / total
    return restricted


# ---------------------------------------------------------------------------
# Tuple loading
# ---------------------------------------------------------------------------

def is_loanword(en_word, lang_words):
    return any(w.lower() == en_word.lower() for w in lang_words.values())


def build_translation_tuples(trans_data):
    raw = []
    all_five = trans_data.get("all_five_tuples", {})
    for en_word, lang_words in all_five.items():
        entry = {"en": en_word}
        entry.update(lang_words)
        raw.append(entry)

    if not raw:
        multi = trans_data.get("multi_lang_tuples", {})
        for en_word, lang_words in multi.items():
            entry = {"en": en_word}
            entry.update(lang_words)
            raw.append(entry)

    tuples = [t for t in raw if not is_loanword(t["en"], {k: v for k, v in t.items() if k != "en"})]
    print(f"  Raw tuples: {len(raw)}, after loanword filter: {len(tuples)} ({len(raw) - len(tuples)} removed)")
    return tuples


# ---------------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------------

def safe_mean(lst):
    return sum(lst) / len(lst) if lst else 0


def probe2_v2(hub, embhub_embedding, baseline_embedding, tokenizer, device, tuples, alpha, baseline_step, k=10):
    random.seed(42)
    n_tuples = len(tuples)

    # ---------------------------------------------------------------
    # Gather word data from EmbHub model
    # ---------------------------------------------------------------
    print("  Computing EmbHub word data...")
    hub_data = {}
    for t_idx, tup in enumerate(tuples):
        for lang in LANGS:
            if lang not in tup:
                continue
            hub_data[(t_idx, lang)] = get_word_embhub_data(
                hub, embhub_embedding, tokenizer, tup[lang], device
            )

    # ---------------------------------------------------------------
    # Gather word data from Baseline model
    # ---------------------------------------------------------------
    print("  Computing Baseline word data...")
    baseline_data = {}
    for t_idx, tup in enumerate(tuples):
        for lang in LANGS:
            if lang not in tup:
                continue
            baseline_data[(t_idx, lang)] = get_word_baseline_emb(
                baseline_embedding, tokenizer, tup[lang], device
            )

    # Token count stats
    single_per_lang = {lang: 0 for lang in LANGS}
    total_per_lang = {lang: 0 for lang in LANGS}
    for (t_idx, lang), wd in hub_data.items():
        total_per_lang[lang] += 1
        if wd["n_tokens"] == 1:
            single_per_lang[lang] += 1
    print(f"  Words per language: {total_per_lang}")
    print(f"  Single-token per language: {single_per_lang}")

    # Alive mask
    all_dists = [wd["anchor_dist"] for wd in hub_data.values()]
    alive_mask = compute_alive_mask(all_dists)
    n_alive = alive_mask.sum().item() if alive_mask is not None else 0
    n_total = alive_mask.shape[0] if alive_mask is not None else 0
    print(f"  Alive anchors: {n_alive}/{n_total} ({n_alive/n_total*100:.1f}%)")

    word_dists_alive = {}
    for key, wd in hub_data.items():
        word_dists_alive[key] = restrict_to_alive(wd["anchor_dist"], alive_mask)

    # ===================================================================
    # TEST A: Anchor distribution overlap
    # ===================================================================
    print("  Test A: Anchor distribution overlap...")

    trans_js_alive = []
    trans_jaccards = []
    pair_data_a = {}

    for t_idx, tup in enumerate(tuples):
        langs_in_tup = [l for l in LANGS if l in tup and (t_idx, l) in hub_data]
        for i, l1 in enumerate(langs_in_tup):
            for l2 in langs_in_tup[i + 1:]:
                d1_alive = word_dists_alive[(t_idx, l1)]
                d2_alive = word_dists_alive[(t_idx, l2)]
                d1_full = hub_data[(t_idx, l1)]["anchor_dist"]
                d2_full = hub_data[(t_idx, l2)]["anchor_dist"]

                js_a = 1.0 - js_divergence(d1_alive, d2_alive)
                j = topk_jaccard(d1_full, d2_full, k)

                trans_js_alive.append(js_a)
                trans_jaccards.append(j)

                pair_key = f"{min(l1,l2)}-{max(l1,l2)}"
                if pair_key not in pair_data_a:
                    pair_data_a[pair_key] = {"js_alive": [], "jaccard": []}
                pair_data_a[pair_key]["js_alive"].append(js_a)
                pair_data_a[pair_key]["jaccard"].append(j)

    rand_js_alive = []
    rand_jaccards = []
    for _ in range(len(trans_js_alive)):
        t1, t2 = random.sample(range(n_tuples), 2)
        l1, l2 = random.sample(LANGS, 2)
        d1 = word_dists_alive.get((t1, l1))
        d2 = word_dists_alive.get((t2, l2))
        if d1 is not None and d2 is not None:
            rand_js_alive.append(1.0 - js_divergence(d1, d2))
            rand_jaccards.append(topk_jaccard(
                hub_data[(t1, l1)]["anchor_dist"],
                hub_data[(t2, l2)]["anchor_dist"], k
            ))

    per_pair_a = {}
    for pair_key, data in sorted(pair_data_a.items()):
        per_pair_a[pair_key] = {
            "js_alive_mean": safe_mean(data["js_alive"]),
            "jaccard_mean": safe_mean(data["jaccard"]),
            "n_pairs": len(data["js_alive"]),
        }

    pval_js_a = pval_j = 1.0
    if trans_js_alive and rand_js_alive:
        _, pval_js_a = stats.mannwhitneyu(trans_js_alive, rand_js_alive, alternative="greater")
    if trans_jaccards and rand_jaccards:
        _, pval_j = stats.mannwhitneyu(trans_jaccards, rand_jaccards, alternative="greater")

    test_a = {
        "n_alive_anchors": int(n_alive),
        "n_total_anchors": int(n_total),
        "translation_js_alive_mean": safe_mean(trans_js_alive),
        "random_js_alive_mean": safe_mean(rand_js_alive),
        "js_alive_gap": safe_mean(trans_js_alive) - safe_mean(rand_js_alive),
        "js_alive_pvalue": pval_js_a,
        "translation_jaccard_mean": safe_mean(trans_jaccards),
        "random_jaccard_mean": safe_mean(rand_jaccards),
        "jaccard_gap": safe_mean(trans_jaccards) - safe_mean(rand_jaccards),
        "jaccard_pvalue": pval_j,
        "per_pair": per_pair_a,
        "n_translation_comparisons": len(trans_js_alive),
        "n_random_comparisons": len(rand_js_alive),
    }

    # ===================================================================
    # TEST B (corrected): EmbHub vs Baseline token embedding cosine
    # Did training *through the hub* shape embeddings so translations
    # are closer than in the baseline?
    # ===================================================================
    print("  Test B: EmbHub vs Baseline embedding cosine (corrected)...")

    trans_cos_embhub = []
    trans_cos_baseline = []
    pair_data_b = {}

    for t_idx, tup in enumerate(tuples):
        langs_in_tup = [l for l in LANGS if l in tup and (t_idx, l) in hub_data]
        for i, l1 in enumerate(langs_in_tup):
            for l2 in langs_in_tup[i + 1:]:
                # EmbHub model's raw token embeddings (alpha=0, no hub contribution)
                cos_embhub = cosine_sim(
                    hub_data[(t_idx, l1)]["raw_emb"],
                    hub_data[(t_idx, l2)]["raw_emb"]
                )
                # Baseline model's raw token embeddings
                cos_baseline = cosine_sim(
                    baseline_data[(t_idx, l1)],
                    baseline_data[(t_idx, l2)]
                )

                trans_cos_embhub.append(cos_embhub)
                trans_cos_baseline.append(cos_baseline)

                pair_key = f"{min(l1,l2)}-{max(l1,l2)}"
                if pair_key not in pair_data_b:
                    pair_data_b[pair_key] = {"cos_embhub": [], "cos_baseline": []}
                pair_data_b[pair_key]["cos_embhub"].append(cos_embhub)
                pair_data_b[pair_key]["cos_baseline"].append(cos_baseline)

    # Random control
    rand_cos_embhub = []
    rand_cos_baseline = []
    for _ in range(len(trans_cos_embhub)):
        t1, t2 = random.sample(range(n_tuples), 2)
        l1, l2 = random.sample(LANGS, 2)
        if (t1, l1) in hub_data and (t2, l2) in hub_data:
            rand_cos_embhub.append(cosine_sim(
                hub_data[(t1, l1)]["raw_emb"],
                hub_data[(t2, l2)]["raw_emb"]
            ))
            rand_cos_baseline.append(cosine_sim(
                baseline_data[(t1, l1)],
                baseline_data[(t2, l2)]
            ))

    per_pair_b = {}
    for pair_key, data in sorted(pair_data_b.items()):
        per_pair_b[pair_key] = {
            "cos_embhub_mean": safe_mean(data["cos_embhub"]),
            "cos_baseline_mean": safe_mean(data["cos_baseline"]),
            "delta": safe_mean(data["cos_embhub"]) - safe_mean(data["cos_baseline"]),
            "n_pairs": len(data["cos_embhub"]),
        }

    # Does EmbHub training increase translation-pair cosine vs baseline?
    pval_embhub_vs_baseline = 1.0
    if trans_cos_embhub and trans_cos_baseline:
        _, pval_embhub_vs_baseline = stats.wilcoxon(
            trans_cos_embhub, trans_cos_baseline, alternative="greater"
        )

    # Translation > random in EmbHub embeddings?
    pval_trans_rand_embhub = 1.0
    if trans_cos_embhub and rand_cos_embhub:
        _, pval_trans_rand_embhub = stats.mannwhitneyu(
            trans_cos_embhub, rand_cos_embhub, alternative="greater"
        )

    # Translation > random in Baseline embeddings?
    pval_trans_rand_baseline = 1.0
    if trans_cos_baseline and rand_cos_baseline:
        _, pval_trans_rand_baseline = stats.mannwhitneyu(
            trans_cos_baseline, rand_cos_baseline, alternative="greater"
        )

    test_b = {
        "baseline_step": baseline_step,
        "trans_cos_embhub_mean": safe_mean(trans_cos_embhub),
        "trans_cos_baseline_mean": safe_mean(trans_cos_baseline),
        "embhub_vs_baseline_delta": safe_mean(trans_cos_embhub) - safe_mean(trans_cos_baseline),
        "pval_embhub_vs_baseline": pval_embhub_vs_baseline,
        "rand_cos_embhub_mean": safe_mean(rand_cos_embhub),
        "rand_cos_baseline_mean": safe_mean(rand_cos_baseline),
        "trans_vs_rand_embhub": safe_mean(trans_cos_embhub) - safe_mean(rand_cos_embhub),
        "pval_trans_rand_embhub": pval_trans_rand_embhub,
        "trans_vs_rand_baseline": safe_mean(trans_cos_baseline) - safe_mean(rand_cos_baseline),
        "pval_trans_rand_baseline": pval_trans_rand_baseline,
        "per_pair": per_pair_b,
        "n_comparisons": len(trans_cos_embhub),
    }

    # ===================================================================
    # TEST C: Contribution vector cosine
    # Are hub contribution vectors more aligned for translation pairs?
    # ===================================================================
    print("  Test C: Contribution vector cosine...")

    trans_cos_contrib = []
    pair_data_c = {}

    for t_idx, tup in enumerate(tuples):
        langs_in_tup = [l for l in LANGS if l in tup and (t_idx, l) in hub_data]
        for i, l1 in enumerate(langs_in_tup):
            for l2 in langs_in_tup[i + 1:]:
                cos_c = cosine_sim(
                    hub_data[(t_idx, l1)]["contribution"],
                    hub_data[(t_idx, l2)]["contribution"]
                )
                trans_cos_contrib.append(cos_c)

                pair_key = f"{min(l1,l2)}-{max(l1,l2)}"
                if pair_key not in pair_data_c:
                    pair_data_c[pair_key] = []
                pair_data_c[pair_key].append(cos_c)

    rand_cos_contrib = []
    for _ in range(len(trans_cos_contrib)):
        t1, t2 = random.sample(range(n_tuples), 2)
        l1, l2 = random.sample(LANGS, 2)
        if (t1, l1) in hub_data and (t2, l2) in hub_data:
            rand_cos_contrib.append(cosine_sim(
                hub_data[(t1, l1)]["contribution"],
                hub_data[(t2, l2)]["contribution"]
            ))

    per_pair_c = {}
    for pair_key, vals in sorted(pair_data_c.items()):
        per_pair_c[pair_key] = {
            "cos_contrib_mean": safe_mean(vals),
            "n_pairs": len(vals),
        }

    pval_contrib = 1.0
    if trans_cos_contrib and rand_cos_contrib:
        _, pval_contrib = stats.mannwhitneyu(
            trans_cos_contrib, rand_cos_contrib, alternative="greater"
        )

    test_c = {
        "trans_cos_contrib_mean": safe_mean(trans_cos_contrib),
        "rand_cos_contrib_mean": safe_mean(rand_cos_contrib),
        "gap": safe_mean(trans_cos_contrib) - safe_mean(rand_cos_contrib),
        "pvalue": pval_contrib,
        "per_pair": per_pair_c,
        "n_comparisons": len(trans_cos_contrib),
    }

    # ===================================================================
    # Examples
    # ===================================================================
    examples = []
    for t_idx in range(min(3, n_tuples)):
        tup = tuples[t_idx]
        ex = {"_en_word": tup.get("en", "?")}
        for lang in LANGS:
            if lang in tup and (t_idx, lang) in hub_data:
                wd = hub_data[(t_idx, lang)]
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
        "test_c_contribution_cosine": test_c,
        "examples": examples,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_results(all_results, output_path, n_tuples):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# Probe 2 v2 — MUSE Cross-Lingual Analysis (Loanword-Filtered, Corrected Test B)\n\n")
        f.write(f"Translation tuples: {n_tuples} (loanword-filtered MUSE pairs)\n\n")
        f.write("Tests:\n")
        f.write("- **Test A**: JS-divergence on alive anchors (primary), top-k Jaccard (secondary)\n")
        f.write("- **Test B** (corrected): EmbHub vs Baseline token embedding cosine\n")
        f.write("  - Asks: did training *through the hub* shape embeddings so translations are closer?\n")
        f.write("- **Test C**: Contribution vector cosine — are hub contributions aligned for translations?\n\n")

        for step, res in all_results.items():
            ta = res["test_a_anchor_overlap"]
            tb = res["test_b_embedding_cosine"]
            tc = res["test_c_contribution_cosine"]

            f.write(f"## Checkpoint step {step}\n\n")
            f.write(f"- Alpha: {res['alpha']}\n")
            f.write(f"- Tuples: {res['n_tuples']}\n")
            f.write(f"- Baseline checkpoint step: {tb['baseline_step']}\n")
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
            f.write(f"\n### Test B — EmbHub vs Baseline Token Embedding Cosine (Corrected)\n\n")
            f.write(f"Baseline checkpoint: step {tb['baseline_step']}\n\n")
            f.write("Did training through the hub produce embeddings where translations are closer?\n\n")
            f.write("| | EmbHub | Baseline | Delta | p-value |\n")
            f.write("|--|--------|---------|-------|--------|\n")
            f.write(f"| Translation pairs | {tb['trans_cos_embhub_mean']:.4f} | {tb['trans_cos_baseline_mean']:.4f} | {tb['embhub_vs_baseline_delta']:+.4f} | {tb['pval_embhub_vs_baseline']:.2e} |\n")
            f.write(f"| Random pairs | {tb['rand_cos_embhub_mean']:.4f} | {tb['rand_cos_baseline_mean']:.4f} | {tb['rand_cos_embhub_mean'] - tb['rand_cos_baseline_mean']:+.4f} | — |\n")
            f.write(f"\n| | Translation | Random | Gap | p-value |\n")
            f.write(f"|--|-----------|--------|-----|--------|\n")
            f.write(f"| EmbHub embs | {tb['trans_cos_embhub_mean']:.4f} | {tb['rand_cos_embhub_mean']:.4f} | {tb['trans_vs_rand_embhub']:+.4f} | {tb['pval_trans_rand_embhub']:.2e} |\n")
            f.write(f"| Baseline embs | {tb['trans_cos_baseline_mean']:.4f} | {tb['rand_cos_baseline_mean']:.4f} | {tb['trans_vs_rand_baseline']:+.4f} | {tb['pval_trans_rand_baseline']:.2e} |\n")

            if tb['per_pair']:
                f.write(f"\nPer language-pair:\n\n")
                f.write("| Pair | N | EmbHub | Baseline | Delta |\n")
                f.write("|------|---|--------|---------|------|\n")
                for pair, vals in sorted(tb['per_pair'].items()):
                    f.write(f"| {pair} | {vals['n_pairs']} | {vals['cos_embhub_mean']:.4f} | {vals['cos_baseline_mean']:.4f} | {vals['delta']:+.4f} |\n")

            # Test C
            f.write(f"\n### Test C — Contribution Vector Cosine\n\n")
            f.write("Are hub contribution vectors more aligned for translation pairs?\n\n")
            f.write("| | Translation | Random | Gap | p-value |\n")
            f.write("|--|-----------|--------|-----|--------|\n")
            f.write(f"| Contribution cosine | {tc['trans_cos_contrib_mean']:.4f} | {tc['rand_cos_contrib_mean']:.4f} | {tc['gap']:+.4f} | {tc['pvalue']:.2e} |\n")

            if tc['per_pair']:
                f.write(f"\nPer language-pair:\n\n")
                f.write("| Pair | N | Contrib cosine |\n")
                f.write("|------|---|---------------|\n")
                for pair, vals in sorted(tc['per_pair'].items()):
                    f.write(f"| {pair} | {vals['n_pairs']} | {vals['cos_contrib_mean']:.4f} |\n")

            # Examples
            if res.get('examples'):
                f.write(f"\nExamples (top-10 anchors):\n\n")
                for ex in res['examples']:
                    en = ex.get("_en_word", "?")
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
    parser = argparse.ArgumentParser(description="Probe 2 v2 — corrected Test B with baseline comparison")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="EmbHub checkpoint paths")
    parser.add_argument("--baseline-dir", required=True,
                        help="Baseline checkpoint directory (e.g. /opt/dlami/nvme/checkpoints/qwen3-0.6b-scratch-baseline)")
    parser.add_argument("--translations", default="temp/frequent_translations.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="temp/probe2_muse_v2_no_loanword_RESULTS.md")
    args = parser.parse_args()

    with open(args.translations) as f:
        trans_data = json.load(f)
    tuples = build_translation_tuples(trans_data)

    if not tuples:
        print("ERROR: No translation tuples found.")
        sys.exit(1)

    print(f"Loaded {len(tuples)} translation tuples")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoints[0])
    all_results = {}

    for ckpt in args.checkpoints:
        step_str = os.path.basename(ckpt).replace("checkpoint-", "")
        target_step = int(step_str)
        print(f"\n{'='*60}")
        print(f"  EmbHub checkpoint: {ckpt} (step {step_str})")
        print(f"{'='*60}")

        # Find matching baseline checkpoint
        baseline_ckpt = find_baseline_checkpoint(args.baseline_dir, target_step)
        if baseline_ckpt is None:
            print(f"  WARNING: No baseline checkpoint found near step {target_step}, skipping.")
            continue
        baseline_step = int(os.path.basename(baseline_ckpt).replace("checkpoint-", ""))
        print(f"  Baseline checkpoint: {baseline_ckpt} (step {baseline_step})")

        # Load both models
        print("  Loading EmbHub model...")
        model_hub, hub, alpha = load_embhub_checkpoint(ckpt, device=args.device)
        embhub_embedding = model_hub.get_input_embeddings()

        print("  Loading Baseline model...")
        model_base = load_baseline_checkpoint(baseline_ckpt, device=args.device)
        baseline_embedding = model_base.get_input_embeddings()

        res = probe2_v2(hub, embhub_embedding, baseline_embedding, tokenizer, device=args.device,
                        tuples=tuples, alpha=alpha, baseline_step=baseline_step)
        all_results[step_str] = res

        del model_hub, hub, model_base
        torch.cuda.empty_cache()

    write_results(all_results, args.output, len(tuples))

    json_path = args.output.replace(".md", ".json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Raw data saved to {json_path}")


if __name__ == "__main__":
    main()
