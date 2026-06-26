# EmbHub Pre-Launch Test Plan

Self-contained plan for verifying the EmbHub layer before committing to the full
multi-day pretraining run. Audience: the Claude instance that wrote the EmbHub init and
training code. Total budget for the required tests: about half a day.

## Background — what was wrong and what we adopted

**Mechanism recap.** EmbHub inserts a layer after the token embedding: each token computes
similarity against N = 1000 learnable anchor embeddings, takes a softmax-weighted sum of
the anchors, and adds it back scaled by a fixed `alpha` (default 0.05). Model: Qwen3-0.6B
architecture, d = 1024, trained from scratch on 30B unique English tokens plus unchanged
target-language portions.

**The problem with the original formula.** The previous implementation used
`softmax(token_emb @ anchors.T / sqrt(d))`. At initialization, every embedding value has
std 0.02 (Qwen3 initializer_range; our anchor init copies it). Walk the arithmetic:

1. Each term of the dot product is (±0.02) x (±0.02) = ±0.0004 with a random sign.
2. Summing 1024 randomly-signed terms cancels heavily: the sum grows like sqrt(1024) = 32,
   not 1024. One score is typically ±0.0004 x 32 ~= ±0.013.
3. The 1000 scores therefore differ from each other by ~0.013. Dividing by sqrt(d) = 32
   squashes the differences to ~0.0004.
4. Softmax weight ratios = exp(difference) = exp(0.0004) ~= 1.0004 → **uniform attention**.
   Every token retrieves the same vector, mean(anchors) — the whole layer collapses to a
   single shared bias.
5. With uniform weights, every anchor receives an identical gradient, so anchors remain
   clones of each other and the uniformity is permanent. The mechanism is born dead and
   training cannot revive it.

This matches what we observed in the previous run (slightly worse PPL, no benefit). Note
the `/sqrt(d)` temperature is correct in standard attention only because Q/K there have
per-dim std ~1 (LayerNorm + projections); raw embeddings are 50x smaller per dim, making
the score spread 2500x smaller than the formula assumes.

**The adopted fix.** Cosine similarity with a learnable temperature (the CLIP pattern;
identical in spirit to QK-Norm, which Qwen3 itself uses inside attention): normalize away
the input scale, then control sharpness with one explicit learnable scalar.

**Why test instead of just launching.** Three reasons:

1. The fix is correct on paper, but its correctness lives in implementation details that
   fail *silently*: normalizing the wrong dim, forgetting to exponentiate the log-scale,
   softmax over the wrong axis, bf16 quirks. These are invisible in the loss curve for
   thousands of steps — and instantly visible in a static check, because we know the
   expected numbers analytically.
2. Init checks cannot see training *dynamics*. Selection must sharpen, anchors must
   specialize rather than collapse into clones or route everything to a few winners. A
   short smoke run catches dynamic failures in hours instead of discovering them days
   into the full run.
3. The previous run had no instrumentation and its checkpoints are lost. The logging
   built for these tests is required infrastructure for the full run — we do not fly
   blind again.

## Reference implementation (the spec being tested)

```python
# selection uses normalized vectors; values are the RAW anchors
q = F.normalize(token_emb, dim=-1)                  # (B, T, d)
k = F.normalize(anchor_embeddings, dim=-1)          # (N, d)
scale = self.log_logit_scale.exp().clamp(max=100)   # scalar
logits = (q @ k.T) * scale                          # (B, T, N)
weights = logits.softmax(dim=-1)
contribution = weights @ anchor_embeddings          # raw anchors as values
output = token_emb + alpha * contribution           # alpha fixed, default 0.05
```

Config details:
- `log_logit_scale`: nn.Parameter, init = log(14) ~= 2.639; exponentiated in forward;
  clamped at exp() <= 100. **Exclude it from weight decay** (same convention as norm
  gains — otherwise weight decay slowly cools the temperature).
- Normalization is for selection only; the weighted sum uses raw, unnormalized anchors.
- No alpha warmup: at init the contribution is ~0.1-0.2% of token norm, so warmup adds
  nothing.
- Anchor init: keep the current `_init_weights` (moment-matched to the reference
  embedding).

## Expected numbers — what "correct" looks like

Softmax weight ratio between two anchors = exp(logit difference). The working window:

| logit spread | weight ratio | regime |
|---|---|---|
| ~0.0004 | 1.0004 | uniform averaging — too cold (the old bug) |
| ~0.3-3 | 1.3-20x | soft selection — the target window |
| ~30 | e^30 | one-hot — too hot (saturated, frozen gradients) |

For the adopted formula at init: cosine of independent random 1024-d vectors has std
1/sqrt(1024) ~= 0.031. Multiplied by scale 14:

- **logit std ~= 0.44** (pass range 0.3-0.6)
- **attention entropy ~= 6.81** vs uniform log(1000) = 6.907 (pass range 6.70-6.88).
  Note: entropy close to uniform at init is correct — the learnable scale sharpens it
  during training. The failure signature is entropy that *stays* there.
- the luckiest of 1000 anchors has cos ~= 0.1 → logit ~= 1.4 → **max attention weight
  typically 3-6x uniform** (~0.003-0.006)

Run all static checks in fp32 (cast if needed) so precision doesn't muddy verdicts; the
smoke run uses the real mixed-precision training config.

---

## Test 1 — Wiring sanity (REQUIRED, minutes)

**Why:** if the EmbHub wrapper changes anything besides adding `alpha * contribution`, the
entire experiment is invalid, and nothing downstream would reveal it. This must hold
before any other number means anything.

**How:** run the same batch through (a) the unmodified base model and (b) the EmbHub model
with `alpha = 0`, both in fp32. Compare final logits elementwise.

**Expect:** max absolute difference < 1e-6. Any larger mismatch = wiring bug; fix before
proceeding.

## Test 2 — Static initialization check (REQUIRED, minutes)

**Why:** verifies the cosine/scale path is actually implemented as specced. Because the
expected statistics are known analytically (table above), any implementation bug shows up
immediately as a number in the wrong place — this is the cheapest possible insurance.

**How:** tokenize a few real batches from the training corpus (~2k-10k tokens — real text,
not random IDs, so token frequencies are realistic). Compute the logits and attention
weights at init. Report: logit std, attention entropy (mean ± std across tokens),
effective anchor count exp(entropy), mean max-weight.

**Expect:** the "Expected numbers" ranges above. Bug signatures if outside them:

| measured | likely cause |
|---|---|
| entropy ~6.90, logit std ~0.08 | log_logit_scale not exponentiated (scale stuck at 2.64) |
| logit std ~0.013 or ~0.0004 | cosine path not active — silent fallback to old dot-product code |
| logit std > 3 at init | scale init or clamp bug |
| NaN/inf | normalize without eps on a zero row (use F.normalize defaults) |
| entropy ~6.907 despite healthy logit std | softmax over the wrong axis |

**Optional (+10 min, recommended):** run the old formula on the same batch and save the
side-by-side table — it becomes the paper's motivation figure ("naive scaled dot-product
retrieval is uniform at initialization").

## Test 3 — Smoke training run (REQUIRED, 500-1000 steps, a few GPU-hours)

**Why:** static checks prove the layer is alive at step 0; they cannot prove it *learns*.
Three dynamic failures are possible even with a correct init: selection never sharpens,
anchors collapse into clones, or routing concentrates on a few anchors while the rest
die. This run catches all three in hours. The logging built here ships with the full run.

**How:** run the real training config with EmbHub, and the no-EmbHub baseline on identical
data, ordering, and seed. Every 50 steps, evaluate on a fixed probe batch and log:

1. attention entropy
2. logit std (variant-agnostic ground truth for selection sharpness)
3. current logit_scale value
4. anchor usage: share of total attention mass on the top-10 anchors, and dead-anchor
   fraction (anchors receiving < 0.1x uniform mass)
5. mean pairwise cosine similarity between anchor vectors (sampled pairs)
6. norm ratio: mean(||alpha * contribution|| / ||token_emb||)
7. train loss / PPL, compared against the baseline run

**Expect (pass criteria):**

1. entropy trends clearly downward from ~6.8; flat above 6.85 through step 1000 = FAIL
2. logit std drifts upward as selection sharpens
3. logit_scale moves away from 14 (either direction — movement means the model uses the
   knob)
4. top-10 anchor mass < ~30%; dead-anchor fraction < ~50% by step 1000 (soft thresholds:
   report the numbers, don't auto-fail)
5. anchor pairwise cosine must NOT trend toward 1 (toward 1 = clone collapse)
6. norm ratio stays < 0.5
7. loss within run-to-run noise of baseline; a persistently growing gap = investigate

## Launch decision rule

- Tests 1 + 2 pass, smoke run meets criteria 1, 2, 5, 7 → **launch the full run the same
  day, same config.** No further deliberation.
- Smoke fails only on usage balance (criterion 4) → add an MoE-style load-balancing
  auxiliary loss, rerun the smoke test only.
- Smoke fails on entropy flatness (criterion 1) → implementation issue in the
  normalize/scale path; return to the Test 2 bug-signature table.

## Test 4 — Gradient symmetry contrast (OPTIONAL, ~10 min)

**Why:** directly demonstrates the clone-trap mechanism for the paper's analysis section.
Not needed for the launch decision.

**How:** one forward+backward on a small batch, for the old formula and the adopted one.
Collect per-anchor gradients; report mean pairwise cosine between them and the
coefficient of variation of per-anchor gradient norms.

**Expect:** old formula → pairwise cosine ~= 1.0, CV ~= 0 (all anchors get the same
update). Adopted formula → cosine clearly < 1, CV > 0.1 (anchors differentiate).

## Deferred to the finetuning experiment (do not run now)

- **Pretrained embedding statistics:** std, per-row norm distribution, per-dim
  anisotropy of the pretrained Qwen3-0.6B embedding matrix. Documents the real s_tok and
  decides whether a raw-dot ablation is even legitimate there.
- **Anchor init comparison:** moment-matching vs sampling 1000 real rows of the
  pretrained embedding table. Hypothesis: row sampling gives meaningful cosines at step 0
  because anchors start inside the same anisotropic subspace as real tokens.

## Future ablations (do NOT bundle into run 1)

- per-dim gains (QK-Norm style) replacing the scalar logit_scale — hypothesis: the router
  learns to downweight language-identity dimensions and route on semantics; the learned
  gains become an interpretability artifact
- alpha sweep 0.01 / 0.05 / 0.1 / 0.2 — the coupling-strength curve (core paper figure)
- num_anchors 100 / 1000 / 10000
- top-k routing; load-balancing loss (if not already added via the decision rule)
- anchor row-sampling init (finetuning setting)

## Deliverables

1. `diagnostics/embhub_diagnostics.py` — runs Tests 1, 2 (and optionally 4); prints a
   markdown results table with PASS/FAIL against the "Expected numbers" section.
2. Training-loop logging hooks for the seven smoke-run metrics (CSV or wandb) — required
   for the full run as well, not test-only code.
3. `diagnostics/RESULTS.md` — measured numbers, one-line verdict per test, and the launch
   decision.
