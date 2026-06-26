# EmbHub Project Notes

Comprehensive notes for context continuity across sessions (this project tends to exhaust the context window, so this file is the single source of truth for a fresh session).

> **Active in-progress task:** see [`docs/CURRENT_TASK.md`](CURRENT_TASK.md). This file is the settled background; that file is the foreground (what we're working on right now, and where to resume if a session is compacted or the machine is switched).

## TL;DR — current state (read this first)

- **What it is:** a learnable "anchor hub" layer inserted after token embeddings; each token retrieves a similarity-weighted mix of shared anchor vectors. Hypothesis: translation- equivalent tokens across languages learn to share anchors, creating cross-lingual alignment without parallel data or code-switching.
- **Architecture status: WORKS mechanically.** After a long detour (see V1/V2), the current recipe (cosine similarity + a high-LR learnable temperature, or a fixed temperature ~50) makes anchor selection sharpen reliably without hurting language-model loss.
- **From-scratch hypothesis: CLEAN NEGATIVE.** Measured three ways with cognate contamination isolated and removed, anchors do NOT meaningfully bootstrap cross-lingual alignment from monolingual data — they mostly encode per-language identity. The genuine non-cognate sharing effect is ~1% and does not grow with training. This is the theoretically expected outcome (monolingual data gives no gradient pressure to align "dog" with "cho").
- **The mid-project benchmark "gains" are NOT real** (paws_de +6.1% etc.). They were recorded while the hub was inert (identical PPL to baseline), have inconsistent sign across sibling tasks, and sit within same-run jitter. Do NOT cite them.
- **Next: FINISH the from-scratch experiments, THEN pivot to finetuning.** Remaining from-scratch work = the per-step Probe 2 refinement (active — see `CURRENT_TASK.md`) + the cheap corrected Test B. Only once those close out does the headline experiment move to the FINETUNING track (add EmbHub to a pretrained multilingual model where cross-lingual structure already exists for the anchors to exploit).
- **Note on claims in this doc:** measured numbers come from the result JSON/CSV files; verdicts ("clean negative", "LR is the lever", "gains are noise") are interpretation grounded in those numbers — solid, but they are the lines to revisit if ever disputed.

## Glossary (terms used throughout)

- **anchor / hub embedding:** one of the N=1000 learnable vectors the layer attends over.
- **alpha:** fixed scalar mixing the hub contribution into the token embedding (`out = token_emb + alpha * contribution`). Default 0.05.
- **logit_scale / temperature:** learnable scalar multiplying cosine similarities before softmax; controls selection sharpness. Stored as `log_logit_scale`, init log(14).
- **entropy:** Shannon entropy of a token's attention over the 1000 anchors. Max (uniform) = log(1000) = 6.907. LOW = sharp/selective, HIGH = uniform/inert.
- **logit_std:** std of a token's pre-softmax scores across anchors; upstream cause of entropy (higher logit_std -> lower entropy). Faster-responding than entropy.
- **norm_ratio:** ||alpha * contribution|| / ||token_emb||; how much the hub perturbs the token. Grows with alpha.
- **dead anchor:** an anchor receiving negligible attention mass. "Per-batch dead" (training callback, ~100 English tokens) is unreliable; "global dead" (Probe 1, full eval set) is the trustworthy version.
- **alive-set Jaccard:** overlap between two languages' sets of used (alive) anchors; measures whether languages share anchors or partition them.
- **smoke callback:** training hook logging 7 diagnostic metrics every 50 steps to wandb.
- **S1..S10:** the ten temperature-revival smoke-test configurations (see the table in the smoke-test section).
- **JS-on-alive-anchors:** 1 - Jensen-Shannon divergence between two words' anchor distributions, restricted to the alive-anchor support; Probe 2's primary metric.

## What This Project Is

EmbHub is a cross-lingual "embedding hub" layer for multilingual LLMs. It inserts a learnable layer after the token embedding that computes a similarity-weighted attention over a set of learnable "anchor" embeddings and adds a scaled contribution back to each token embedding.

The layer sits between the embedding and transformer layers:
```
token_ids -> Embedding -> EmbHub -> Transformer layers -> LM head
```

**The hypothesis.** Semantically equivalent tokens across languages (e.g. "dog" / "cho" / "Hund") should learn to retrieve similar anchor subsets. That shared retrieval would pull their representations into a common subspace, improving cross-lingual transfer — conceptually similar to what code-switching achieves, but without requiring mixed-language data or parallel corpora, and with a continuously tunable coupling strength (alpha) that can be set to 0 at inference to recover a normal model.

**Why it might beat code-switching (the original motivation).** Code-switching forces direct token-to-token cross-lingual attention, which (a) needs manually constructed mixed-language data and (b) makes the model prone to generating mixed-language output. EmbHub instead routes all languages through a *shared* anchor bank: tokens attend to the same anchors rather than to each other, a gentler, removable form of coupling.

**Relationship to prior work.** Closest is PREALIGN (EMNLP 2024), same goal but requires a bilingual dictionary (GPT-4-generated) for contrastive pre-alignment + input-only code- switching; EmbHub needs no parallel data. Also related: soft-prompt / prefix tuning (EmbHub differs in being similarity-based per-token retrieval, applied during pretraining, for cross- lingual alignment rather than task adaptation) and MoE shared experts (anchors are MoE-like but soft-selected over all anchors, not hard top-k).

**Two intended experimental settings:**
1. **From-scratch** (run so far): train Qwen3-0.6B from scratch on multilingual data with EmbHub. The HARD setting — anchors must *create* cross-lingual alignment from monolingual data, where no gradient directly forces "dog" and "cho" together.
2. **Finetuning a pretrained multilingual model** (not yet run): add EmbHub to a model that already encodes "dog" ~= "cho", so anchors only *exploit and amplify* existing alignment. The more favorable setting; the planned next phase, but only AFTER the from-scratch experiments finish (see Timeline conclusion).

Model architecture: Qwen3-0.6B (d=1024, 28 layers, RoPE — so the anchor layer operates on pure token embeddings before any positional information, which is correct for this design).

## Timeline of What We Tried

Read as a narrative: each stage motivated the next. The headline conclusion at the end supersedes earlier optimism — the benchmark "gains" recorded mid-project turned out to be noise, and the frozen-temperature problem has been SOLVED, not left open.

### V1 — Scaled Dot-Product (failed: layer born dead)

**Formula:** `softmax(token_emb @ anchors.T / sqrt(d))`

**Config:** 1000 anchors, alpha=0.05, d=1024. Data: 10B English + 5x100M others (vi, zh, ru, de, ar), 3 epochs ~= 30B total tokens (the same 10B English duplicated 3x). 8x H200, global batch 512, ~30K steps. First AWS machine; checkpoints later lost.

**Result:** EmbHub ~0.4 PPL WORSE than baseline at step 10K. Machine lost before 20K/30K eval.

**Diagnosis (later confirmed):** the formula produces UNIFORM attention at initialization. With Qwen3 initializer_range=0.02, each dot-product term is ~(0.02)^2 and the 1024-term sum (random signs) is ~0.013; dividing by sqrt(1024)=32 squashes score differences to ~0.0004. Softmax over differences that tiny is uniform -> every token retrieves mean(anchors), a single shared bias vector. Worse, uniform weights mean every anchor gets an IDENTICAL gradient, so anchors stay clones and uniformity is self-locking. The sqrt(d) temperature is correct in standard attention only because Q/K there have per-dim std ~1 (post-LayerNorm + projections); raw embeddings are ~50x smaller per dim, so scores are ~2500x outside the temperature's design range. The layer was born dead.

**Code:** V1 ran on the *same* source files as V2 (`hub_layer_v2.py`, `model_wrapper_v2.py`, `test_embhub_v2.py`) — the V1 dot-product formula was overwritten in place when V2 began, so V1's exact code is not preserved as a separate file. See `CHANGELOG.md` (v1 section) + git history for the V1 formula/config.

### V2 — Cosine + Learnable Temperature (current architecture)

**Formula:**
```python
q = F.normalize(token_embeddings, dim=-1)
k = F.normalize(self.hub_embeddings, dim=-1)
scale = self.log_logit_scale.exp().clamp(max=100.0)
logits = (q @ k.T) * scale
weights = logits.softmax(dim=-1)
hub_contribution = weights @ self.hub_embeddings   # raw (unnormalized) anchors as values
output = token_embeddings + alpha * hub_contribution
```

Rationale: cosine removes the embedding-scale problem (cosine of random 1024-d vectors has std ~= 0.031 regardless of norm); the learnable temperature (init log(14), CLIP-style) then sets selection sharpness explicitly. Same normalize-then-scale pattern as QK-Norm, which Qwen3 itself uses inside attention. Selection uses normalized vectors; values use raw anchors (so an anchor can grow its norm to contribute more).

**Pre-launch diagnostics (all passed):** wiring (alpha=0 == base model); static init (logit_std=0.44, entropy=6.81, effective_anchors~=908 — soft but differentiated, as predicted); gradient differentiation (per-anchor grad cosine 0.93 vs v1's 1.0 — anchors no longer in the clone trap).

**V2 full training config:** Data: 30B English + 5x300M others, 1 epoch, ~31.5B UNIQUE tokens (no repetition — a deliberate upgrade from v1's duplicated data). 8x H200, global batch 512, ~31.5K steps. LR 3e-4 cosine_with_min_lr (min_lr_rate=0.1), warmup 500, weight decay 0.1, betas (0.9, 0.95), grad clip 1.0. Ratio is ~1:100 English:per-language (30B vs 300M each), vs v1's 10B:100M.

**V2 first full run — the hub was INERT (selection never sharpened):** Smoke-callback metrics over the full run showed the temperature FROZEN at 14.02 (init 14.0): entropy drifted UP toward uniform (6.81 -> 6.87), logit_std peaked ~0.65 then fell to 0.32, norm_ratio grew 0.002 -> 0.026, dead_anchor_fraction stayed ~0. Consequence: hub and baseline had IDENTICAL perplexity at every checkpoint (differences < 0.04 in all six languages). The hub acted as a near-constant bias; the 596M base model did all the work.

Final-checkpoint PPL (baseline vs EmbHub):

| Language | Baseline | EmbHub |
|---|---|---|
| en | 19.25 | 19.24 |
| ru | 15.46 | 15.45 |
| vi | 16.41 | 16.42 |
| de | 22.61 | 22.62 |
| ar | 19.30 | 19.28 |
| zh | 72.99 | 72.95 |

**V2 anchor-geometry analysis (post-training) — ruled out the WRONG explanation:** A natural misread is "anchors collapsed into clones." Direct measurement disproves it: residual pairwise cosine = -0.001 (anchors near-orthogonal in their individuating directions), PCA of residuals flat (PC1/PC2 = 1.0, top-10 = 3.3% of variance). The raw pairwise cosine of ~0.527 was entirely a SHARED MEAN OFFSET (anchors = shared mu + diverse residual), which is benign. So the problem was never collapse — the temperature never rose, so selection stayed uniform. (Script: `diagnostics/check_anchor_geometry.py`.)

**Root cause of the frozen temperature (diagnosed, then FIXED in the smoke tests below):**
- Weight decay 0.1 on `log_logit_scale`: decay pull (~0.1 * 2.64 * lr) was ~14x larger than the gradient (~5.7e-6) and pointed toward zero. HF Trainer auto-excludes bias/norm params from decay but NOT "log_logit_scale".
- Genuinely tiny gradient: near-uniform softmax is nearly flat in the scale direction (chicken-and-egg: need sharper selection to get a scale gradient, need scale to sharpen).
- A bf16-save bug was initially suspected as a third cause but is a RED HERRING for the freeze — mixed precision keeps fp32 master weights that accumulate, so quantization would staircase, not freeze. The `.float()`-on-save fix is still correct and was kept; it just was not the cause.

**Code:** `hub_layer_v2.py` (the layer), `model_wrapper_v2.py` (inject/save/load/remove), `test_embhub_v2.py` (unit tests); training via `train.py` + `scripts/train_qwen3_0.6b.sh` (baseline `…_baseline.sh`) with live metrics from `diagnostics/smoke_callback.py`; pre-launch checks `diagnostics/embhub_diagnostics.py`; post-training geometry `diagnostics/check_anchor_geometry.py`.

### V2 temperature-revival smoke tests (10 arms, ~1000-1500 steps each) — temperature SOLVED

Tested two knobs (weight decay on/off x LR multiplier 1x/75x), plus init and a fixed-vs- learnable temperature option, crossed with alpha. Arm configs:

| Arm | temperature | WD on scale | LR mult | init | alpha |
|---|---|---|---|---|---|
| S1 | learnable | off | 1x | 14 | 0.05 |
| S2 | learnable | on | 75x | 14 | 0.05 |
| S3 | learnable | off | 75x | 14 | 0.05 |
| S4 | learnable | off | 75x | 30 | 0.05 |
| S5 | fixed 30 | — | — | — | 0.05 |
| S6 | fixed 50 | — | — | — | 0.05 |
| S7 | fixed 50 | — | — | — | 0.10 |
| S8 | fixed 50 | — | — | — | 0.20 |
| S9 | learnable | off | 75x | 14 | 0.10 |
| S10 | learnable | off | 75x | 14 | 0.20 |

(Fixed-temperature arms hold the scale at the value in the "temperature" column; "init" applies only to learnable arms.)

Key findings:
- **LR is the real lever, NOT weight-decay removal.** S1 (decay off, LRx 1) did NOT move (scale stayed 14.02). Every LRx-75 arm moved hard. This CORRECTED the original "decay is the bug" hypothesis — removing decay alone was insufficient; the dominant problem was the tiny gradient, which a high LR overcomes.
- Best learnable arm S3 (decay off, LRx 75, init 14, alpha 0.05): scale climbed from 14 to the mid-50s (~58 in the 10-arm smoke run; ~54.6 in a longer S3 re-run, where it then held flat), entropy fell 6.8 -> ~3.2, loss matched baseline. The temperature converges well under the clamp of 100 — a real equilibrium, so after ~step 1100 S3 effectively behaves like a fixed- temperature run (~scale 54-58).
- Fixed-temperature arms (scale 30/50) also work and remove the learnable-scalar failure class entirely. (S5 at scale 30 sharpens less: entropy ~5.5.)
- Alpha: 0.05 is right; 0.10 borderline-OK; 0.20 measurably hurts LM loss (S8/S10 loss ~4.3 vs ~3.9; norm_ratio grows with alpha, so the contribution starts corrupting the token).
- Dead anchors: arms that sharpen too fast (high fixed scale, high init) kill anchors faster. A longer S3 re-run (to 3400 steps) shows the per-100-English-token dead fraction from the training callback climbing 0.3% (1500) -> 14% (2500) -> 27% (3400) while train loss KEEPS IMPROVING (3.92 -> 3.46) — i.e. consolidation does not hurt LM loss. (This per-batch callback number is unreliable in absolute terms; Probe 1 re-measures it properly on the eval set.)

**Two phases (to avoid step-count confusion):** the 10-arm screen above ran only ~1000–1500 steps each (enough to see the temperature move). S3 was then re-run much longer, producing the checkpoints the probes below use (1500 / 3250 / 5500, now extending to 6500). So "smoke arm" step counts and "probe checkpoint" step counts refer to different runs.

**Code:** `run_smoke_tests.py` (the 10 arm configs + sequential launcher), `smoke_train.py` (the instrumented training script each arm runs).

### V2 anchor probes on S3 checkpoints (1500 / 3250 / 5500) — the decisive experiment

Two probes on the (now sharply-selecting) S3 checkpoints. Both reproduce the forward selection in fp32 under no_grad and analyze the attention WEIGHTS. NOTE: the training callback's "dead anchor" metric is computed on only ~100 ENGLISH tokens and is unreliable (100 tokens cannot exercise 1000 anchors); the probes re-measure properly over the full 10M-per-language eval set. (Scripts: `anchor_probes.py`; results `anchor_probes_RESULTS.*`, `probe2_muse_*`.)

**Probe 1 — global & per-language anchor usage:**
- Anchor consolidation is REAL (not a probe artifact): global dead fraction 0% (1500) -> 47.1% (3250) -> 44.5% (5500); dead-in-EVERY-language 0 -> 110 -> 125 (genuinely wasted). Alive-in- all-languages 563 -> 122 -> 137. Plateaus ~45%, i.e. the model effectively uses ~550 of 1000 anchors. (Motivates an anchor-count ablation: 1000 is more than needed.)
- Cross-lingual structure DEGRADES as consolidation proceeds: per-language alive-set Jaccard collapses (en-zh 0.62 -> 0.16 -> 0.18; en-ru 0.90 -> 0.31 -> 0.34). High early overlap was trivial (almost nothing dead yet). End state: anchors PARTITION BY LANGUAGE/SCRIPT (en-de stays highest, en-zh lowest) — they encode language identity more than shared meaning.

**Probe 2 — cross-lingual anchor overlap (translation vs frequency-matched random):** Ran in three increasingly-clean versions (a 4th, per-step, is in progress); the cleanup MATTERED:
1. Random word list (first attempt): no signal — but the word list was broken (single-token filter left vi/ru/de with ~6-8 words). Inconclusive, not negative.
2. Frequency-filtered MUSE tuples + mean-pooling (5044 tuples): apparent POSITIVE — JS-on- alive-anchors gap +0.069 / +0.078 / +0.076 across 1500/3250/5500, all p~=0.
3. **Cognate/loanword removed (1716 tuples)** — the trustworthy version. Removing pairs where the "translation" was identical to the English word (e.g. Vietnamese entry "love" -> "love") COLLAPSED the gap ~20x: JS gap +0.0041 / +0.0042 / +0.0038 (1500/3250/5500). Still p- significant at n=25,740 (p = 6e-5 / 3.5e-3 / 1.25e-2) but only a ~0.7% relative lift. The earlier positive was MOSTLY cognates trivially sharing anchors (same token -> same anchors), not cross-lingual transfer.
4. **Per-step frequency lists (in progress).** Restrict each language's frequent-word list to the tokens the model had ACTUALLY trained on by each checkpoint (extends the checkpoint set to 6500), instead of full-corpus frequencies — so the word list matches what the model could have learned by that step. Pipeline + status in `CURRENT_TASK.md`.

So the genuine (non-cognate) cross-lingual anchor-sharing effect is REAL BUT VERY WEAK, dominated by per-language identity structure, consistent across checkpoints, and NOT strengthening with training. The "dog/flame/czech" qualitative examples confirm it: an occasional genuinely-shared anchor (e.g. anchors 104/279 for en/de "flame") sits atop a strong per-language frequency/script signature (anchors 432, 655, 811, 311 fire for almost everything in zh/ru/ar regardless of meaning).

**Caveat — Probe 2 "Test B" was mis-measured and must be re-run.** Test B (post-hub embedding cosine, with-hub vs without-hub) as run is NOT informative: it compared `e` vs `e + 0.05*contribution` within the SAME model (i.e. 1.0*e vs 1.05*e), so it only measured the trivial 5% nudge (and showed the hub helping random pairs as much as translations — an artifact). The CORRECT Test B compares the EmbHub model's trained embeddings vs the BASELINE model's trained embeddings (both alpha=0), or compares contribution-VECTOR similarity for translation vs random pairs. It has NOT yet been run and is the one remaining clean confirmation for the from-scratch setting.

**Code:** `diagnostics/anchor_probes.py` (Probe 1 + base Probe 2); Probe 2 MUSE variants `anchor_probe2_muse.py` → `anchor_probe2_muse_no_loan_word.py` (the trustworthy v3) → `anchor_probe2_muse_v2{,_no_loan_word}.py`. The in-progress per-step v4 (item 4 above) uses `decode_token_ids.py` → `count_words_from_text.py` → `build_frequent_translations.py` (commands in `CURRENT_TASK.md`).

### Conclusion / current status (supersedes earlier optimism)

- The architecture WORKS mechanically: cosine + high-LR temperature makes selection sharpen reliably without destabilizing LM loss (best recipe: S3 ~= fixed scale ~54-58, alpha 0.05).
- The from-scratch cross-lingual HYPOTHESIS is a CLEAN NEGATIVE (measured three ways, cognate contamination isolated and removed): anchors do NOT meaningfully bootstrap cross-lingual alignment from monolingual data — they mostly learn per-language identity, with only a ~1% non-cognate sharing effect that does not grow with training. This is the theoretically expected outcome (monolingual data gives no gradient pressure to align translations).
- The mid-project benchmark "gains" (paws_de +6.1%, xnli_vi +2.1%, etc.) are NOT real signal. Recorded while the hub was inert (identical PPL to baseline), inconsistent sign across sibling tasks, the largest on PAWS at its ~0.5 chance line, and within same-run checkpoint- to-checkpoint jitter. Do NOT cite them. (McNemar on paired predictions is the right tool if revisited.)

**Next steps (priority order) — finetuning comes only AFTER the from-scratch experiments are finished:**
1. FINISH from-scratch. (a) Per-step Probe 2 refinement — frequency lists restricted to tokens actually seen by each checkpoint (active; see `CURRENT_TASK.md`). (b) The CORRECTED Test B (EmbHub-vs-baseline trained embeddings; and/or contribution-vector similarity) as the final clean confirmation — cheap, no training, not expected to overturn the negative. (c) Optional: anchor-count ablation (256/512 vs 1000, since ~45% are wasted) and, ONLY if sharper selection is wanted, a load-balancing aux loss — but note loss IMPROVED during anchor death, so balancing may be unnecessary or even harmful unless the probe shows it helps cross-lingual overlap specifically.
2. THEN pivot the headline experiment to the FINETUNING track — add EmbHub to a pretrained multilingual model, where "dog" ~= "cho" already exists for the anchors to exploit. This is where the idea has a real mechanism; a clean from-scratch negative + a finetuning positive is a stronger, more honest paper than a borderline from-scratch claim.

---

**Note on the `**Code:**` references above.** Code files are edited **in place** across stages — they are NOT snapshotted per stage. So if the same file is listed under two stages, that file was *updated between them* and now reflects the LATER stage; the earlier stage's logic is no longer in the current file. To see an earlier stage's actual code, use `CHANGELOG.md` or git history, not the current file. (Example: `hub_layer_v2.py` held both the V1 dot-product formula and the V2 cosine+temperature formula — the V1 version was overwritten when V2 began.)

## Code Structure

```
cross_lingual_embeddings_hub/
├── hub_layer_v2.py          # EmbHub nn.Module (cosine + learnable temp) — CURRENT architecture
├── model_wrapper_v2.py      # inject / save / load / remove EmbHub on a HF model
├── train.py                 # Main training script (HF Trainer; baseline + EmbHub)
├── smoke_train.py           # train.py variant for the S1–S10 smoke arms; adds --save_token_ids
│                            #   (dumps ids_rank{N}.jsonl + meta.json for the per-step word analysis)
├── run_smoke_tests.py       # Runs the 10 smoke arms sequentially (8 GPUs each), stops at --stop-at-step
├── prepare_data.py          # Download + sample CulturaX (train+eval); deterministic per SEED
├── run_clm.py               # HF reference script (comparison only)
│
├── eval/                    # Evaluation pipeline
│   ├── ppl.py               # Core PPL (sliding window)
│   ├── benchmarks.py        # Core benchmarks (lm-eval wrapper)
│   ├── eval_checkpoint.py   # Single checkpoint: load once, run PPL + benchmarks
│   └── eval_parallel.py     # Queue-based parallel launcher across GPUs
│
├── diagnostics/
│   │  # -- training-time + post-training geometry --
│   ├── smoke_callback.py                  # Training callback (7 metrics every 50 steps → wandb)
│   ├── embhub_diagnostics.py              # Tests 1, 2, 4 from the test plan
│   ├── check_anchor_geometry.py           # Post-training anchor geometry (the "collapse?" check)
│   ├── embhub_diagnostics_every250.csv    # Extracted smoke-run metrics
│   │  # -- anchor probes (Probe 1 + Probe 2) --
│   ├── anchor_probes.py                   # Probe 1 (global/per-lang usage) + Probe 2 (base)
│   ├── anchor_probe2_muse.py              # Probe 2 with MUSE translation pairs
│   ├── anchor_probe2_muse_no_loan_word.py # ↑ cognate/loanword-filtered (the TRUSTWORTHY version)
│   ├── anchor_probe2_muse_v2.py           # Probe 2 v2 (per web-reviewer recommendations)
│   ├── anchor_probe2_muse_v2_no_loan_word.py # ↑ v2, loanword-filtered
│   │  # -- per-step "words actually seen by step S" pipeline (feeds Probe 2) --
│   ├── decode_token_ids.py                # ids_rank{N}.jsonl → text_rank{N}.jsonl  [env: embeddings_hub]
│   ├── count_words_from_text.py           # text → per-lang word counts per step (fasttext)  [env: fasttext_env]
│   ├── find_frequent_words.py             # full-corpus word frequencies per language (non-per-step)
│   ├── build_frequent_translations.py     # frequent words → MUSE translation pairs for Probe 2
│   └── embhub_diagnostics_test_plan_after_tried_version_1_test_results.md
│
├── scripts/
│   ├── train_qwen3_0.6b.sh                # EmbHub training launch
│   ├── train_qwen3_0.6b_baseline.sh       # Baseline training launch
│   └── extract_diagnostics_log_from_wandb.py
│
├── configs/                 # Custom GPT-2 configs (future experiments; unused by the Qwen3 runs)
│   ├── gpt2-base.json
│   ├── gpt2-medium.json
│   └── gpt2-large.json
│
├── docs/
│   ├── PROJECT_NOTES.md     # This file — single source of truth across sessions
│   ├── embhub_smoke_tests.md                              # S1–S10 smoke-test details
│   ├── version_1.md                                       # V1 history
│   └── embhub_diagnostics_test_plan_after_tried_version_1.md
│
├── temp/                    # GITIGNORED — all results land here; do NOT survive a fresh clone
│   ├── anchor_probes_RESULTS.{json,md}        # Probe 1 results
│   ├── probe2_muse*_RESULTS.{json,md}         # Probe 2 results (incl. no-loanword + v2)
│   ├── frequent_words.json / frequent_translations.json
│   ├── muse_dicts/                            # downloaded MUSE bilingual dictionaries
│   └── claude_web_review_probe2_rerun.md      # web-Claude's Probe 2 review
│
├── data/                    # GITIGNORED — CulturaX raw parquet + sampled train/eval (see Data Pipeline)
│
├── test_embhub_v2.py        # V2 unit tests (26, all passing)
├── CHANGELOG.md             # V1 vs V2 history
├── .gitignore
│
├── hub_layer.py             # V1 (deprecated, reference only)
├── model_wrapper.py         # V1 (deprecated)
└── test_embhub.py           # V1 tests (deprecated)
```

> Note: `smoke_train.py` and `train.py` share most logic; `smoke_train.py` is the instrumented fork used by the smoke harness (adds the smoke-arm flags + `--save_token_ids`).

### ⚠️ What is NOT in git (read if you are a fresh session / on a different dev machine)

`temp/` and `data/` are **gitignored**. A fresh clone — or switching to a different dev machine — starts with **neither**, so any path under them referenced in this doc will be absent until regenerated. Nothing here is lost work; it just isn't version-controlled.

| Missing on a fresh machine | How to get it back |
|---|---|
| `temp/*_RESULTS.{json,md}` (Probe 1 / Probe 2 numbers) | re-run the probe scripts (needs checkpoints + eval set) |
| `temp/frequent_words.json`, `temp/frequent_translations.json` | re-run `find_frequent_words.py` / `build_frequent_translations.py` |
| `temp/muse_dicts/` | re-download MUSE bilingual dictionaries |
| `temp/lid.176.bin` (fasttext language-ID model) | auto-downloads on first `count_words_from_text.py` run |
| `data/` (CulturaX raw parquet + sampled train/eval) | `prepare_data.py download sample` (see Data Pipeline) |
| **Training checkpoints** (`embhub.pt`, HF checkpoints) | **NOT regenerable without re-training — must be backed up to S3/root disk** (see Machine Setup) |

So: the *measured numbers quoted in this doc are the durable record*; the result files backing them may not exist on the machine you are reading this from.

## Data Pipeline

**Download + sample:** `python prepare_data.py download sample --tokenizer-name Qwen/Qwen3-0.6B --num-workers 4` (`--num-workers` is just the download parallelism knob.)

Data layout:
```
data/
├── raw/                          # Downloaded parquet files from CulturaX
│   ├── en/ (50 files)            # ~43B tokens worth; ~35 used to hit the 30B target
│   ├── vi/ (5 files)             # each lang hit its 300M target in ~1 file; 5 = safety margin
│   ├── zh/ (5 files)
│   ├── ru/ (5 files)
│   ├── de/ (5 files)
│   └── ar/ (5 files)
└── Qwen_Qwen3-0.6B/
    ├── train/{ar,de,en,ru,vi,zh}/   # 30B en + 5×300M others
    └── eval/{ar,de,en,ru,vi,zh}/    # 10M per language
```

`num_files` per language lives in `LANG_CONFIG` in `prepare_data.py`. It was cut from 300/90/200/… to **50 en / 5 each** (the original run downloaded ~2.7 TB but used only ~40 files); the new plan is ~175 GB. The sampler stops at the token target, so the extra files are just margin.

**Reproducibility caveat (important if re-creating the dataset).** File selection is deterministic given `SEED=42` and a stable CulturaX file listing — BUT `random.sample(pop, k)` is **not nested**: `sample(k=50)` is *not* the first 50 of `sample(k=300)`. So this 50/5 plan is a **different dataset than the original 300/90/200 run** — you cannot cheaply reproduce the old training data by shrinking `num_files`. For a fresh start that doesn't matter; if exact reproduction is ever needed, pin the filenames (`python prepare_data.py --dry-run … > data_file_plan.txt` and commit it) rather than relying on the seed alone.

Train and eval are sampled sequentially from the same deterministic document ordering (seed=42). First N documents -> train, next M documents -> eval. Zero overlap. (The 10M-per-language eval split is also what Probe 1 / Probe 2 use for proper global anchor-usage and cross-lingual measurements.)

## Training

Both scripts use `accelerate launch` with 8 GPUs, DDP, bf16 mixed precision.

```bash
bash scripts/train_qwen3_0.6b_baseline.sh  # baseline
bash scripts/train_qwen3_0.6b.sh            # embhub
```

Key training code details:
- `SaveEmbHubCallback`: saves `embhub.pt` + `embhub_config.json` at each checkpoint (rank 0 only via `args.should_save`).
- `EmbHubSmokeCallback`: logs 7 diagnostic metrics every 50 steps to wandb (embhub runs only). IMPORTANT: these metrics are computed on a fixed ~100-English-token probe batch, so the usage-based ones (dead_anchor_fraction, top10_anchor_mass) are unreliable in absolute terms — use the eval-set probes for those. The per-token metrics (entropy, logit_std, norm_ratio) and the scalar logit_scale are fine.
- Resume: auto-detects last checkpoint via `get_last_checkpoint`, also respects `--resume_from_checkpoint`.
- Data preprocessing uses `main_process_first` + HF datasets `.map()` caching.
- `--ddp_timeout 21600` needed because data preprocessing for 31.5B tokens exceeds the default 30-minute DDP timeout.

**bf16 save bug (fixed in code).** `save_embhub` originally called `torch.save(hub.state_dict(), ...)`, saving in the model's bf16 dtype. Fix applied: `{k: v.float() for k, v in hub.state_dict().items()}`. NOTE: this was a real correctness fix for saved checkpoints, but it was NOT the cause of the frozen temperature (see Timeline — that was weight decay + tiny gradient, fixed by removing decay and raising the scale LR).

**Temperature fix to apply in the next training run** (from the smoke tests): exclude `log_logit_scale` from weight decay AND give it a high LR (~75x base) in its own param group; or make the temperature a fixed constant (~50) and drop the learnable scalar entirely. Without this the hub stays inert (the original V2 run's failure mode).

## Evaluation

```bash
# Single checkpoint (loads model once, runs PPL then benchmarks)
python eval/eval_checkpoint.py --checkpoint path/to/ckpt --eval-dir data/Qwen_Qwen3-0.6B/eval --bf16

# All checkpoints in parallel (queue-based, up to 8 GPUs, no idle waiting)
python eval/eval_parallel.py --eval-dir data/Qwen_Qwen3-0.6B/eval --bf16

# PPL only / benchmarks only
python eval/eval_parallel.py --eval-dir data/Qwen_Qwen3-0.6B/eval --bf16 --ppl-only
python eval/eval_parallel.py --bf16 --bench-only
```

PPL uses sliding-window strategy from HF docs (stride = block_size // 2). Token counting uses `(target_ids[:, 1:] != -100).sum()` — fixes a bug in the HF reference code where `num_valid_tokens - batch_size` undercounts for non-first windows.

Benchmarks: XNLI (6 langs), Belebele (6 langs), XCOPA (2 langs), XStoryCloze (4 langs), PAWS-X (3 langs), HellaSwag (5 langs) = 26 tasks total. All `output_type: multiple_choice` (log- likelihood scoring, correct for base models, no chat template). Auto-patches 3 lm-eval dataset paths (`xnli` -> `facebook/xnli`, `xcopa` -> `cambridgeltl/xcopa`, `paws-x` -> `google-research-datasets/paws-x`) for huggingface_hub 1.x compatibility.

CAUTION on interpreting benchmarks at this scale: a 0.6B model on ~30B tokens sits near chance on many of these (XNLI non-English ~0.33 = 3-way chance; HellaSwag ~0.27; PAWS ~0.5). Small deltas between runs are usually noise (see Timeline conclusion). Use McNemar on paired predictions for significance, and prioritize the eval-set anchor probes + PPL over raw benchmark deltas.

**Important:** Model must be on GPU before passing to `HFLM` — `HFLM` ignores the `device` kwarg for pre-initialized models and reads `model.device`. The old eval code had this bug (ran benchmarks on CPU). Fixed in `eval/eval_checkpoint.py`.

Eval needs `lm-eval`, which should be in a separate conda env (`eval`) to avoid dependency conflicts with PyTorch.

## Machine Setup

**Dev machine (Claude Code session):** code development and testing. No GPU — tests run on CPU. The dev machine is **interchangeable / disposable** (it has been swapped repeatedly); only what is committed to git survives a switch (see "What is NOT in git" above). Current one is at `/data/thuat_2/projects/cross_lingual_embeddings_hub/`, 32 CPUs.

**Conda envs (apply to both dev and training machines):**
- `embeddings_hub` — torch 2.12.0 (cu130), transformers 5.9.0, datasets 4.8.5, accelerate 1.13.0. Used for training and for `decode_token_ids.py`.
- `eval` — lm-eval + dependencies (separate to avoid PyTorch version conflicts).
- `fasttext_env` — fasttext + numpy 1.x (no transformers). Used ONLY for `count_words_from_text.py` (needs `temp/lid.176.bin`). Kept separate **on purpose**: fasttext language-ID runs under `multiprocessing`, and importing transformers in the same process tree risks a fork-after-threads deadlock. So decode (transformers) and count (fasttext) are two scripts in two envs by design.

**Training machine:** AWS instance with 8x H200 (141GB each), 192 CPUs, CUDA 13.2.
- Checkpoints on `/opt/dlami/nvme/` (ephemeral DLAMI instance store, 28TB) — data survives reboot on standard EC2 but DLAMI may reformat on reboot.
- Root disk: 2.9TB only.
- ⚠️ **Checkpoints are the one irreplaceable artifact and are NOT in git.** Back up milestone checkpoints to S3 or root disk immediately. V1's checkpoints were lost this way, and dev/training machines have been lost repeatedly since — assume any uncommitted, un-backed-up state is temporary.

**Recurring infrastructure issues:**
- Fabric Manager version mismatch: `cuda-toolkit-13-0` installs FM 610 but driver is 595. Fix:
```
sudo apt-get install --allow-downgrades nvidia-fabricmanager=595.71.05-1ubuntu1
sudo apt-mark hold nvidia-fabricmanager
sudo systemctl restart nvidia-fabricmanager
sudo systemctl status nvidia-fabricmanager
```
- `unattended-upgrades` overrides the hold and re-upgrades FM. Fix: `sudo systemctl disable unattended-upgrades`.
- NCCL NVLS can fail after FM issues. `NCCL_NVLS_ENABLE=0` is a safe workaround (affects only transport, not computational results). Currently set in the embhub training script only (baseline ran without it — both are valid).

## Wandb

Project: `cross_lingual_embedding_hub`
Key runs:
- `qwen3-0.6b-scratch-baseline` — baseline v2 training.
- `qwen3-0.6b-hub1000-a0.05` — embhub v2 training (has `embhub/*` diagnostic panels). This is the INERT run (frozen temperature); the temperature-revival smoke tests (S1-S10) and the longer S3 re-run are separate.

## Git

Repository: `git@github.com:nguyenhuuthuat09/cross_lingual_embeddings_hub.git`
Branch: `main`
