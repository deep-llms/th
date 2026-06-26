"""Probe 2 with MUSE-based translation pairs (replaces hardcoded 80 tuples).

Uses all_five_tuples from build_frequent_translations.py (words that have
translations in all 5 non-English languages), plus the en→xx MUSE pairs.

Run on the training machine where checkpoints + data live.

Usage:
  python diagnostics/anchor_probe2_muse.py \
      --checkpoints /path/to/checkpoint-1500 /path/to/checkpoint-3250 \
      --translations temp/frequent_translations.json \
      --output temp/probe2_muse_RESULTS.md
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
NON_EN_LANGS = ["vi", "zh", "ru", "de", "ar"]


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
    return model, hub


def get_anchor_weights(hub, token_embeddings):
    with torch.no_grad():
        q = F.normalize(token_embeddings.float(), dim=-1)
        k = F.normalize(hub.hub_embeddings.float(), dim=-1)
        scale = hub.log_logit_scale.float().exp().clamp(max=100.0)
        logits = (q @ k.T) * scale
        weights = logits.softmax(dim=-1)
    return weights


def get_word_anchor_dist(hub, embedding, tokenizer, word, device):
    ids = tokenizer(word, add_special_tokens=False, return_tensors="pt")["input_ids"]
    ids = ids.to(device)
    with torch.no_grad():
        token_emb = embedding(ids)
    weights = get_anchor_weights(hub, token_emb)
    return weights.squeeze(0).mean(dim=0), ids.shape[1]


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


def build_translation_tuples(trans_data):
    """Build (en, {lang: word}) tuples from frequent_translations.json.

    Sources:
    1. all_five_tuples: en words with MUSE translations in all 5 non-English langs
    2. en_xx_pairs: en→xx pairs (may cover more words, but not all 5 langs)

    Returns list of dicts: {"en": ..., "vi": ..., "zh": ..., ...}
    """
    tuples = []

    # Source 1: all-5-language tuples (best for cross-lingual comparison)
    all_five = trans_data.get("all_five_tuples", {})
    for en_word, lang_words in all_five.items():
        entry = {"en": en_word}
        entry.update(lang_words)
        tuples.append(entry)

    if tuples:
        print(f"  All-5-lang tuples: {len(tuples)}")
        return tuples

    # Fallback: multi-lang tuples (>=2 languages)
    multi = trans_data.get("multi_lang_tuples", {})
    for en_word, lang_words in multi.items():
        entry = {"en": en_word}
        entry.update(lang_words)
        tuples.append(entry)

    print(f"  Multi-lang tuples (>=2): {len(tuples)}")
    return tuples


def probe2_muse(hub, embedding, tokenizer, device, tuples, k=10):
    random.seed(42)

    # Get anchor distributions for all words
    word_dists = {}  # (tuple_idx, lang) -> distribution
    token_counts = {}
    skipped = 0

    for t_idx, tup in enumerate(tuples):
        for lang in LANGS:
            if lang not in tup:
                continue
            word = tup[lang]
            dist, n_tokens = get_word_anchor_dist(hub, embedding, tokenizer, word, device)
            word_dists[(t_idx, lang)] = dist
            token_counts[(t_idx, lang)] = n_tokens

    # Token count stats
    single_per_lang = {lang: 0 for lang in LANGS}
    total_per_lang = {lang: 0 for lang in LANGS}
    for (t_idx, lang), n in token_counts.items():
        total_per_lang[lang] += 1
        if n == 1:
            single_per_lang[lang] += 1
    print(f"  Words per language: {total_per_lang}")
    print(f"  Single-token per language: {single_per_lang}")

    # Translation pair similarities
    trans_jaccards = []
    trans_js_sims = []
    pair_data = {}  # lang_pair -> list of (jaccard, js_sim)

    for t_idx, tup in enumerate(tuples):
        langs_in_tup = [l for l in LANGS if l in tup and (t_idx, l) in word_dists]
        for i, l1 in enumerate(langs_in_tup):
            for l2 in langs_in_tup[i + 1:]:
                w1 = word_dists[(t_idx, l1)]
                w2 = word_dists[(t_idx, l2)]
                j = topk_jaccard(w1, w2, k)
                js = 1.0 - js_divergence(w1, w2)
                trans_jaccards.append(j)
                trans_js_sims.append(js)

                pair_key = f"{min(l1,l2)}-{max(l1,l2)}"
                if pair_key not in pair_data:
                    pair_data[pair_key] = {"trans_j": [], "trans_js": []}
                pair_data[pair_key]["trans_j"].append(j)
                pair_data[pair_key]["trans_js"].append(js)

    # Random control: different tuples, same language pairs
    rand_jaccards = []
    rand_js_sims = []
    n_tuples = len(tuples)
    for _ in range(len(trans_jaccards)):
        t1, t2 = random.sample(range(n_tuples), 2)
        l1, l2 = random.sample(LANGS, 2)
        w1 = word_dists.get((t1, l1))
        w2 = word_dists.get((t2, l2))
        if w1 is not None and w2 is not None:
            rand_jaccards.append(topk_jaccard(w1, w2, k))
            rand_js_sims.append(1.0 - js_divergence(w1, w2))

    # Per-pair means
    per_pair = {}
    for pair_key, data in sorted(pair_data.items()):
        per_pair[pair_key] = {
            "jaccard_mean": sum(data["trans_j"]) / len(data["trans_j"]),
            "js_sim_mean": sum(data["trans_js"]) / len(data["trans_js"]),
            "n_pairs": len(data["trans_j"]),
        }

    # Significance
    if trans_jaccards and rand_jaccards:
        _, pval_j = stats.mannwhitneyu(trans_jaccards, rand_jaccards, alternative="greater")
        _, pval_js = stats.mannwhitneyu(trans_js_sims, rand_js_sims, alternative="greater")
    else:
        pval_j = pval_js = 1.0

    # Examples (first 3 tuples)
    examples = []
    for t_idx in range(min(3, n_tuples)):
        tup = tuples[t_idx]
        ex = {"en": tup.get("en", "?")}
        for lang in LANGS:
            if lang in tup and (t_idx, lang) in word_dists:
                w = word_dists[(t_idx, lang)]
                ex[lang] = {
                    "word": tup[lang],
                    "top_anchors": w.topk(k).indices.tolist(),
                    "n_tokens": token_counts[(t_idx, lang)],
                }
        examples.append(ex)

    mean_trans_j = sum(trans_jaccards) / len(trans_jaccards) if trans_jaccards else 0
    mean_rand_j = sum(rand_jaccards) / len(rand_jaccards) if rand_jaccards else 0
    mean_trans_js = sum(trans_js_sims) / len(trans_js_sims) if trans_js_sims else 0
    mean_rand_js = sum(rand_js_sims) / len(rand_js_sims) if rand_js_sims else 0

    return {
        "n_tuples": n_tuples,
        "n_translation_comparisons": len(trans_jaccards),
        "n_random_comparisons": len(rand_jaccards),
        "translation_jaccard_mean": mean_trans_j,
        "random_jaccard_mean": mean_rand_j,
        "jaccard_gap": mean_trans_j - mean_rand_j,
        "jaccard_pvalue": pval_j,
        "translation_js_sim_mean": mean_trans_js,
        "random_js_sim_mean": mean_rand_js,
        "js_gap": mean_trans_js - mean_rand_js,
        "js_pvalue": pval_js,
        "per_pair": per_pair,
        "examples": examples,
        "single_token_per_lang": single_per_lang,
        "total_per_lang": total_per_lang,
    }


def write_results(all_results, output_path, n_tuples):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# Probe 2 — MUSE-based Cross-Lingual Anchor Overlap\n\n")
        f.write(f"Translation tuples: {n_tuples} (from MUSE dictionaries, frequency-filtered)\n\n")

        for step, p2 in all_results.items():
            f.write(f"## Checkpoint step {step}\n\n")
            f.write(f"- Translation tuples used: {p2['n_tuples']}\n")
            f.write(f"- Translation comparisons: {p2['n_translation_comparisons']}\n")
            f.write(f"- Random comparisons: {p2['n_random_comparisons']}\n\n")

            f.write(f"Words per language: {p2['total_per_lang']}\n")
            f.write(f"Single-token per language: {p2['single_token_per_lang']}\n\n")

            f.write("| Metric | Translation | Random | Gap | p-value |\n")
            f.write("|--------|------------|--------|-----|--------|\n")
            f.write(f"| Top-10 Jaccard | {p2['translation_jaccard_mean']:.4f} | {p2['random_jaccard_mean']:.4f} | {p2['jaccard_gap']:.4f} | {p2['jaccard_pvalue']:.2e} |\n")
            f.write(f"| 1 - JS divergence | {p2['translation_js_sim_mean']:.4f} | {p2['random_js_sim_mean']:.4f} | {p2['js_gap']:.4f} | {p2['js_pvalue']:.2e} |\n")

            if p2['per_pair']:
                f.write(f"\nPer language-pair:\n\n")
                f.write("| Pair | N pairs | Jaccard | JS Sim |\n")
                f.write("|------|---------|---------|--------|\n")
                for pair, vals in sorted(p2['per_pair'].items()):
                    f.write(f"| {pair} | {vals['n_pairs']} | {vals['jaccard_mean']:.4f} | {vals['js_sim_mean']:.4f} |\n")

            if p2.get('examples'):
                f.write(f"\nExamples (top-10 anchors):\n\n")
                for ex in p2['examples']:
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
    parser = argparse.ArgumentParser(description="Probe 2 with MUSE translation pairs")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--translations", default="temp/frequent_translations.json",
                        help="Output from build_frequent_translations.py")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="temp/probe2_muse_RESULTS.md")
    args = parser.parse_args()

    # Load translation tuples
    with open(args.translations) as f:
        trans_data = json.load(f)
    tuples = build_translation_tuples(trans_data)

    if not tuples:
        print("ERROR: No translation tuples found. Run build_frequent_translations.py first.")
        print("  (Make sure find_frequent_words.py ran on the training machine with data)")
        sys.exit(1)

    print(f"Loaded {len(tuples)} translation tuples")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoints[0])
    all_results = {}

    for ckpt in args.checkpoints:
        step = os.path.basename(ckpt).replace("checkpoint-", "")
        print(f"\n{'='*60}")
        print(f"  Checkpoint: {ckpt} (step {step})")
        print(f"{'='*60}")

        model, hub = load_checkpoint(ckpt, device=args.device)
        embedding = model.get_input_embeddings()

        p2 = probe2_muse(hub, embedding, tokenizer, args.device, tuples)
        all_results[step] = p2

        del model, hub
        torch.cuda.empty_cache()

    write_results(all_results, args.output, len(tuples))

    json_path = args.output.replace(".md", ".json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Raw data saved to {json_path}")


if __name__ == "__main__":
    main()
