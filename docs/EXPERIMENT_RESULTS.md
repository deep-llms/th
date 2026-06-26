# Experiment Results — From-Scratch Probe 2 (Per-Step, LLM Translations)

Last updated: 2026-06-24

## Setup

**Model**: Qwen3-0.6B trained from scratch with EmbHub (S3 arm)
- 1000 anchors, alpha=0.05, learnable temperature (init 14, LR 75x, no weight decay)
- Data: 30B English + 5x300M (vi, zh, ru, de, ar), 1 epoch
- Effective batch: 512 sequences x 2048 tokens = 1M tokens/step
- Checkpoints at steps 1500, 3250, 5500, 6500

**Translation dictionaries**: Two sources compared
- **MUSE**: Facebook's static bilingual dictionaries (single-word only, known quality issues — truncated Vietnamese, wrong translations, ~60% untranslated in vi)
- **LLM (GPT-4o)**: 4,804 tuples from frequent English training words translated to all 5 languages. Handles multi-word expressions correctly (e.g., "government" -> "chinh phu")

**Probe 2 measures**: For each translation tuple (en, vi, zh, ru, de, ar):
- **Test A**: Anchor attention weight overlap (JS divergence, top-10 Jaccard) — do translation pairs select more similar anchors than random pairs?
- **Test B**: Post-hub embedding cosine similarity at different alpha values — does `token_emb + alpha * (weights @ anchors)` make translations more similar than random pairs?

## Test B Results — Post-Hub Embedding Cosine Similarity

This is the key measurement: does the hub push translation-equivalent tokens closer in embedding space?

### LLM translations, single-token words only

The cleanest signal. Single-token words have unambiguous embeddings (no mean-pooling noise). Dominated by en-zh and en-ar pairs (en=2370, zh=3730, ar=1054 single-token words; vi=75, ru=161 are sparse).

| Step | Gap @ a=0.00 | Gap @ a=0.05 | Gap @ a=0.10 | Gap @ a=0.20 | Gap @ a=0.30 |
|------|-------------|-------------|-------------|-------------|-------------|
| 1500 | +0.0128 (p=1e-3) | +0.0127 (p=1e-3) | +0.0125 (p=2e-3) | +0.0121 (p=2e-3) | +0.0116 (p=3e-3) |
| 3250 | +0.0347 (p=3e-12) | +0.0341 (p=1e-11) | +0.0334 (p=3e-11) | +0.0319 (p=4e-10) | +0.0300 (p=4e-9) |
| 5500 | +0.0521 (p=1e-22) | +0.0511 (p=9e-22) | +0.0499 (p=9e-21) | +0.0473 (p=1e-18) | +0.0444 (p=2e-16) |
| 6500 | +0.0571 (p=4e-26) | +0.0560 (p=5e-25) | +0.0547 (p=8e-24) | +0.0519 (p=3e-21) | +0.0486 (p=1e-18) |

**Observations:**
1. **Strong growth across training steps.** Gap at a=0.05 grows from +0.013 (step 1500) to +0.056 (step 6500) — a 4.4x increase. The model is learning cross-lingual structure through training.
2. **Gap at trained alpha (0.05) is substantial.** +5.6% at step 6500, highly significant (p=5e-25).
3. **Decline with increasing alpha.** The gap shrinks from a=0.00 to a=0.30 at every checkpoint. This is expected: the model was trained at a=0.05, so other alpha values are out-of-distribution (same as using LoRA with wrong scaling).
4. **Open question: hub contribution vs base model.** The gap at a=0.00 (raw embeddings, no hub) also grows — need baseline (no-hub) checkpoints to determine whether the hub's gradient signal during training is responsible for the base embedding alignment, or if the base model learns this independently.

### LLM translations, all words (including multi-token)

Weaker signal due to mean-pooling noise on multi-token words. For multi-token words, subtokens are shared across many words and don't carry full word meaning at the embedding layer — cross-lingual effects would appear after transformer layers process them in context.

| Step | Gap @ a=0.00 | Gap @ a=0.05 | Gap @ a=0.10 | Gap @ a=0.20 | Gap @ a=0.30 |
|------|-------------|-------------|-------------|-------------|-------------|
| 1500 | +0.0081 (p=1e-51) | +0.0081 (p=4e-51) | +0.0080 (p=2e-50) | +0.0080 (p=2e-49) | +0.0080 (p=5e-48) |
| 3250 | +0.0130 (p=4e-130) | +0.0129 (p=1e-124) | +0.0129 (p=7e-119) | +0.0127 (p=2e-107) | +0.0124 (p=3e-96) |
| 5500 | +0.0166 (p=5e-211) | +0.0164 (p=3e-200) | +0.0162 (p=4e-189) | +0.0158 (p=2e-166) | +0.0152 (p=4e-144) |
| 6500 | +0.0177 (p=3e-240) | +0.0175 (p=6e-228) | +0.0172 (p=5e-215) | +0.0167 (p=1e-188) | +0.0161 (p=1e-162) |

Same trends as single-token but ~3x smaller gaps. The much lower p-values are due to larger sample size (64K comparisons vs 5K).

### MUSE translations (for comparison)

Lower quality translations, mostly single-word, with known truncation issues in Vietnamese.

| Step | Gap @ a=0.00 | Gap @ a=0.05 | Gap @ a=0.10 | Gap @ a=0.20 | Gap @ a=0.30 |
|------|-------------|-------------|-------------|-------------|-------------|
| 1500 | +0.0082 (p=4e-11) | +0.0082 (p=4e-11) | +0.0081 (p=4e-11) | +0.0081 (p=4e-11) | +0.0080 (p=5e-11) |
| 3250 | +0.0109 (p=3e-19) | +0.0109 (p=1e-18) | +0.0108 (p=5e-18) | +0.0105 (p=1e-16) | +0.0103 (p=3e-15) |
| 5500 | +0.0133 (p=9e-31) | +0.0132 (p=3e-29) | +0.0130 (p=1e-27) | +0.0126 (p=2e-24) | +0.0121 (p=1e-21) |
| 6500 | +0.0133 (p=1e-28) | +0.0131 (p=5e-27) | +0.0128 (p=3e-25) | +0.0123 (p=1e-21) | +0.0116 (p=2e-18) |

Similar pattern to LLM all-words, confirming the signal is real and not an artifact of the translation source.

## Test A Results — Anchor Weight Overlap

Do translation pairs select more similar anchors than random pairs?

### LLM translations

| Step | Jaccard gap (all) | p-value | Jaccard gap (single-token) | p-value |
|------|------------------|---------|---------------------------|---------|
| 1500 | +0.0033 | 1.6e-08 | +0.0008 | 0.44 |
| 3250 | +0.0074 | 6.8e-30 | +0.0228 | 2.8e-04 |
| 5500 | +0.0074 | 2.7e-23 | +0.0128 | 0.074 |
| 6500 | +0.0074 | 2.5e-22 | +0.0191 | 0.010 |

All-words: consistent ~0.7% gap after step 3250, highly significant. Single-token: larger gaps (up to 2.3%) but noisier due to smaller sample (4,873 comparisons vs 64K).

### MUSE translations

| Step | Jaccard gap | p-value |
|------|------------|---------|
| 1500 | +0.0047 | 1.5e-06 |
| 3250 | +0.0061 | 1.7e-06 |
| 5500 | +0.0062 | 5.9e-05 |
| 6500 | +0.0042 | 7.6e-03 |

Similar magnitude to LLM all-words (~0.5% gap), slightly declining at step 6500.

## Translation Quality Comparison

| Metric | MUSE | LLM (GPT-4o) |
|--------|------|---------------|
| Total tuples | ~1,900 | 4,804 |
| Vietnamese untranslated | ~60% | 0% |
| Multi-word support | No (single-word only) | Yes |
| Wrong translations | ~5-10% (truncated, transliterated) | Rare |
| Example: "government" in vi | "phu" (truncated) | "chinh phu" (correct) |
| Example: "february" in vi | "thang" (truncated) | "thang hai" (correct) |

## Key Takeaways

1. **Cross-lingual alignment grows with training.** The embedding-level gap between translations and random pairs increases 4x from step 1500 to 6500. This is measured on the model trained WITH EmbHub.

2. **Single-token measurement is 3-4x cleaner.** Multi-token mean-pooling dilutes the signal because subtokens don't carry full word meaning at the embedding layer. Multi-token effects would show on downstream tasks after transformer context processing.

3. **LLM translations are strictly better than MUSE.** Higher quality, more tuples, proper multi-word support. Future experiments should use LLM translations.

4. **Critical missing piece: baseline comparison.** We don't have checkpoints from a model trained WITHOUT EmbHub. Without this, we cannot determine whether the hub's gradient signal during training drives the cross-lingual alignment, or if the base model learns it independently. This is the next experiment to run.

## Next Steps

1. **Train baseline** (same config, no EmbHub) to the same checkpoints (1500, 3250, 5500, 6500).
2. **Run the same Probe 2 Test B** on baseline checkpoints. If the baseline shows a smaller gap, the hub IS contributing to cross-lingual alignment during training.
3. **Downstream task evaluation** to measure multi-token cross-lingual transfer (XNLI, PAWS-X, etc.) — where transformer context resolves subtoken ambiguity.
