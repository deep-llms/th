# Experiment Results: Baseline vs S3 α=0.15 vs S3 α=0.20

All models: Qwen3-0.6B trained from scratch, 8x H200 (141GB each), effective batch 512 sequences (2048 tokens each), 1M tokens/step.
S3 config: scale_no_wd=True, scale_lr_mult=75, scale_init=14, num_hub_embeddings=1000.

## 1. Training Loss

| Step | Baseline | S3 α=0.15 | S3 α=0.20 |
|------|----------|-----------|-----------|
| 1500 | 3.907 | 3.898 | 3.898 |
| 3250 | 3.478 | 3.479 | 3.478 |
| 5500 | 3.302 | 3.304 | 3.303 |
| 6500 | 3.251 | 3.253 | 3.253 |

All three models have nearly identical loss throughout training. EmbHub does not hurt LM performance.

## 2. EmbHub Metrics (S3 variants only)

### Step 1500

| Metric | S3 α=0.15 | S3 α=0.20 |
|--------|-----------|-----------|
| Entropy | 4.200 | 4.441 |
| Logit Std | 2.555 | 2.381 |
| Logit Scale | 54.60 | 52.10 |
| Top-10 Anchor Mass | 0.219 | 0.196 |
| Dead Anchor Frac | 0.000 | 0.000 |
| Anchor Pairwise Cos | 0.031 | 0.029 |
| Norm Ratio | 0.039 | 0.051 |

### Step 3250

| Metric | S3 α=0.15 | S3 α=0.20 |
|--------|-----------|-----------|
| Entropy | 3.023 | 3.734 |
| Logit Std | 2.594 | 2.590 |
| Logit Scale | 54.60 | 52.92 |
| Top-10 Anchor Mass | 0.368 | 0.324 |
| Dead Anchor Frac | 0.034 | 0.002 |
| Anchor Pairwise Cos | 0.201 | 0.191 |
| Norm Ratio | 0.071 | 0.083 |

### Step 5500

| Metric | S3 α=0.15 | S3 α=0.20 |
|--------|-----------|-----------|
| Entropy | 3.110 | 4.106 |
| Logit Std | 2.161 | 2.144 |
| Logit Scale | 54.60 | 51.29 |
| Top-10 Anchor Mass | 0.307 | 0.284 |
| Dead Anchor Frac | 0.022 | 0.001 |
| Anchor Pairwise Cos | 0.319 | 0.293 |
| Norm Ratio | 0.077 | 0.090 |

### Step 6500

| Metric | S3 α=0.15 | S3 α=0.20 |
|--------|-----------|-----------|
| Entropy | 3.136 | 4.077 |
| Logit Std | 2.010 | 1.926 |
| Logit Scale | 54.60 | 50.50 |
| Top-10 Anchor Mass | 0.323 | 0.301 |
| Dead Anchor Frac | 0.016 | 0.000 |
| Anchor Pairwise Cos | 0.343 | 0.317 |
| Norm Ratio | 0.079 | 0.093 |

Key observations:
- α=0.15 has lower entropy (more concentrated anchor usage) than α=0.20
- α=0.20 has higher norm ratio (hub contribution is proportionally larger)
- α=0.15 has slightly higher anchor pairwise cosine (anchors collapsing slightly more)
- α=0.20 has almost zero dead anchors; α=0.15 develops ~1.6-3.4% dead anchors

## 3. Probe Test A — Anchor Weight Overlap (S3 variants only)

Translation pairs should share more top-k anchors than random pairs if EmbHub learns cross-lingual structure.

### Top-10 Jaccard

| Step | S3 α=0.15 Trans | S3 α=0.15 Rand | S3 α=0.15 Gap | S3 α=0.20 Trans | S3 α=0.20 Rand | S3 α=0.20 Gap |
|------|-----------------|----------------|---------------|-----------------|----------------|---------------|
| 1500 | 0.0683 | 0.0691 | -0.0008 | 0.0575 | 0.0588 | -0.0013 |
| 3250 | 0.1211 | 0.1168 | +0.0043 | 0.0968 | 0.0901 | +0.0067 |
| 5500 | 0.1162 | 0.1090 | +0.0072 | 0.0986 | 0.0898 | +0.0088 |
| 6500 | 0.1089 | 0.1057 | +0.0031 | 0.0981 | 0.0893 | +0.0089 |

### 1 - JS Divergence

| Step | S3 α=0.15 Trans | S3 α=0.15 Rand | S3 α=0.15 Gap (p-value) | S3 α=0.20 Trans | S3 α=0.20 Rand | S3 α=0.20 Gap (p-value) |
|------|-----------------|----------------|-------------------------|-----------------|----------------|-------------------------|
| 1500 | 0.6085 | 0.6092 | -0.0007 (4.12e-01) | 0.6372 | 0.6363 | +0.0009 (1.76e-01) |
| 3250 | 0.5697 | 0.5626 | +0.0071 (5.26e-02) | 0.6079 | 0.5988 | +0.0092 (1.57e-02) |
| 5500 | 0.5642 | 0.5588 | +0.0055 (1.56e-01) | 0.6169 | 0.6040 | +0.0129 (1.16e-03) |
| 6500 | 0.5657 | 0.5570 | +0.0087 (6.19e-02) | 0.6186 | 0.6051 | +0.0135 (**1.03e-03**) |

α=0.20 shows stronger and more significant anchor overlap than α=0.15. At step 6500, α=0.20 reaches p=1.03e-03 on JS similarity.

## 4. Probe Test B — Post-Hub Embedding Cosine Similarity

Measures cosine similarity of translation pairs vs random pairs at the embedding level (alpha=0.0 = raw embeddings, no hub contribution).

### At alpha=0.0 (raw embeddings, comparable to baseline)

| Step | Baseline Gap | S3 α=0.15 Gap | S3 α=0.20 Gap |
|------|-------------|---------------|---------------|
| 1500 | +0.0117 (4.34e-03) | +0.0119 (3.41e-03) | +0.0125 (2.22e-03) |
| 3250 | +0.0312 (3.88e-09) | +0.0286 (1.78e-08) | +0.0285 (2.41e-08) |
| 5500 | +0.0465 (5.84e-16) | +0.0418 (2.58e-14) | +0.0421 (1.91e-14) |
| 6500 | +0.0504 (7.38e-18) | +0.0457 (2.36e-16) | +0.0462 (1.66e-16) |

All models develop cross-lingual structure in raw embeddings. Baseline has slightly larger gap, suggesting EmbHub models offload some cross-lingual signal to the hub instead of the base embeddings.

### Effect of adding hub contribution (all test alphas, step 6500)

| Test α | S3 α=0.15 Gap | S3 α=0.20 Gap |
|--------|---------------|---------------|
| 0.00 | +0.0457 | +0.0462 |
| 0.05 | +0.0454 | +0.0460 |
| 0.10 | +0.0450 | +0.0457 |
| 0.20 | +0.0441 | +0.0451 |
| 0.30 | +0.0430 | +0.0442 |

For both models, the gap is highest at α=0.0 (raw embeddings) and decreases as hub contribution is added. The hub contribution pushes both translation and random pairs in a similar direction, slightly reducing the cross-lingual gap rather than amplifying it. Baseline gap at α=0.0 is +0.0504 for reference.

## 5. Summary

1. **Loss**: All three models are equivalent. EmbHub adds no perplexity cost.
2. **Anchor overlap (Test A)**: α=0.20 > α=0.15. Higher alpha produces more statistically significant cross-lingual anchor sharing (p=1.03e-03 at step 6500).
3. **Embedding similarity (Test B)**: Baseline slightly beats both EmbHub variants on raw embedding cosine gap (+0.0504 vs ~0.046). The anchors do learn cross-lingual structure (Test A shows significant sharing), but adding hub contribution increases similarity for both translation and random pairs, so the gap does not widen at the embedding level. The effect may appear downstream after transformer layers process the hub-modified embeddings.
4. **EmbHub health**: α=0.20 keeps anchors healthier (0% dead) while α=0.15 develops ~1.6% dead anchors. α=0.20 has higher entropy (more diverse anchor usage).
5. **At 6500 steps** (early training), the hub contribution is small (norm_ratio ~8-9%). The cross-lingual benefits of EmbHub may become more apparent later in training as the hub grows stronger.
