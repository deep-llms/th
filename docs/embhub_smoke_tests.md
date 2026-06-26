# EmbHub Temperature Revival — Smoke Tests

Run all arms in parallel (one per GPU). Each is a short smoke test; no decision tree. Goal: find which configuration makes the temperature climb and selection sharpen, so the hub stops acting as a near-uniform constant bias.

## Background — what is already settled (do NOT re-investigate or "fix")

These are settled by direct measurement on the 31.5K-step checkpoint. Several plausible-sounding fixes target problems that the data shows do not exist — each is listed with the number that rules it out, so we don't waste GPUs on them.

**The temperature is the bug.** `log_logit_scale` was frozen at its init (2.64 = log 14) for the entire 21.75K-step logged run. Root cause: weight decay 0.1 produces a pull (~0.1 * 2.64 * lr) about **14x larger** than its gradient (~5.7e-6), pointing toward zero. Decay overpowers the signal, the temperature can't rise, softmax stays near-uniform, and the hub collapses to a constant bias added to every token. This is why hub and baseline ended with identical perplexity — the mechanism was inert.

**Anchors are NOT collapsed.** Residual pairwise cosine = **-0.001** (anchors are near-orthogonal in their individuating directions); PCA spectrum is flat (PC1/PC2 = 1.0, top-10 singular values explain only 3.3% of variance). The anchors are 1000 *distinct* directions sharing a common offset vector. The "0.51 raw pairwise cosine" seen during training is entirely that shared offset, which is a benign thing for the layer to learn — NOT clones. => Do NOT add a diversity / MoE load-balancing loss to "prevent collapse." There is no collapse. Dead-anchor fraction was 0, so there are also no dead anchors for load-balancing to revive. This is purely a temperature problem.

**bf16 is a red herring for the freeze.** Mixed precision keeps fp32 master weights that *accumulate* across steps, so quantization would cause a staircase, not a freeze. A dead-flat value is the decay-vs-tiny-gradient signature, not a dtype artifact. => Keep the `.float()` on save (it is correct on its own merits), but do NOT treat dtype as the cause or chase fp32-everywhere hacks to fix the freeze.

**alpha is not the suspect for the freeze.** hub_embeddings learned fine (they formed the shared-offset-plus-diverse-residual structure), so gradients reach the hub. alpha scales the selection-path and value-path gradients *equally*, so it cannot single out and starve one specific parameter like the temperature. => Do NOT raise alpha expecting it to fix the temperature. (alpha is still worth sweeping for a different reason — see the alpha arms — but it is not a temperature fix, and raising it weakens the "removable / low-risk at inference" property, so 0.05 stays the headline setting.)

**Gradients differentiated correctly at init.** Per-anchor gradient cosine was 0.93 (not 1.0) with the cosine+temperature formula, confirming anchors were never in the identical-gradient "clone trap" that the old dot-product/sqrt(d) formula produced. The mechanism's plumbing is right; only the temperature knob was held down.

General rule for proposing any change: name the measured number that motivates it. "Remove weight decay from the scalar" has a number (14x decay-over-gradient). "Add diversity loss" does not (residual cosine 0, dead fraction 0). Fix the thing with evidence.

## How many iterations

**1000 steps per arm.** The temperature responds within tens to low-hundreds of steps once decay/LR are fixed, but `logit_std` and `entropy` lag (in the original run logit_std took ~2500 steps to peak even while inert). 500 steps shows whether the scale *starts* moving; 1000 also shows logit_std rising and entropy beginning to bend, which is what distinguishes "knob is now free" from "knob moves but selection still isn't sharpening." If a run is ambiguous at 1000, extend that one to 2000 rather than re-running.

## Logging for every arm

- Log every **50 steps** (250 is too coarse to see an early slope).
- Use a **fixed probe batch**, identical across all steps and all arms, so metrics compare.
- You do NOT need to re-run the baseline. Every knob in these arms (weight decay on `log_logit_scale`, the LR multiplier on `log_logit_scale`, fixed vs learnable temperature, and alpha) lives entirely inside the EmbHub layer. None of them exist in a no-EmbHub model, so the existing `qwen3-0.6b-scratch-baseline` run is a valid control for all arms. Use its already-known loss trajectory (loss ~6.4 over the first ~1000 steps, declining) as the instability reference: if an arm's loss diverges noticeably from that, an aggressive scale-LR is destabilizing training.
- Confirm the **step-0 probe** reproduces the static-init numbers (logit_std ~0.44, entropy ~6.81). If step 0 doesn't match, the logging/precision path is wrong — fix that before reading anything else.

Metrics to log each probe: `log_logit_scale`, `logit_std`, `entropy`, `top10_anchor_mass`, `dead_anchor_frac`, `anchor_pairwise_cos` (residual), `norm_ratio`, `train/loss`.

## What to look at (and why this matters)

The mechanism is a chain: **scale rises -> logit_std rises -> entropy falls.** The links respond at different speeds, so look at them in this order. Reading only entropy at 1000 steps can make a working fix look like a failure, because entropy is the slowest link.

| metric | responds in | what it tells you |
|---|---|---|
| `log_logit_scale` | tens of steps | is the temperature free to move at all (the direct test of the decay/LR fix) |
| `logit_std` | low hundreds | is scale movement translating into sharper, more differentiated selection |
| `entropy` | ~1000+ steps | end result: selection becoming non-uniform (slowest; expect it to only *begin* falling by 1000) |
| `anchor_pairwise_cos` (residual) | slow | guard: must NOT trend toward 1 (would mean anchors merging) |
| `train/loss` vs baseline | continuous | guard: aggressive scale-LR shouldn't destabilize training |

A learnable-temperature run is "working" if scale moves AND logit_std rises (entropy starting to bend is a bonus, not required at 1000 steps). For the fixed-temperature arms there is no scale to move, so judge those directly by logit_std/entropy.

> Keep this section: the natural instinct is to grade these runs on entropy, and entropy is exactly the metric that lags. Without this ordering you risk discarding a successful arm at 1000 steps.

## Arms to run

Two temperature knobs (weight decay on/off, LR multiplier) plus init and a fixed-scale option, crossed with an alpha lever. Naming: WD = weight decay on log_logit_scale, LRx = LR multiplier on log_logit_scale, init = starting scale value, alpha = hub mixing.

Learnable-temperature arms (alpha = 0.05):
- **S1** — WD off, LRx 1, init 14 (isolates the decay fix alone)
- **S2** — WD on, LRx 75, init 14 (tests whether brute LR can overpower decay)
- **S3** — WD off, LRx 75, init 14 (principled fix: remove headwind + overcome small grad)
- **S4** — WD off, LRx 75, init 30 (adds sharper start to escape the flat-gradient region)

Fixed-temperature arms (log_logit_scale is a constant, not a parameter — no grad, no optimizer entry, no decay), alpha = 0.05:
- **S5** — fixed scale 30
- **S6** — fixed scale 50

Why the fixed arms exist: a single learnable scalar was never a contribution, and it has already cost a full run plus multiple debug cycles. Fixing it removes the entire optimizer/decay/precision failure class in one line; learnable temperature then becomes an ablation footnote rather than a dependency.

Alpha sweep (0.05 is small; you want to test larger). Carry it on the **fixed scale 50** setting so alpha is the ONLY variable — on a learnable-temperature arm a behavior change could come from either alpha or the temperature moving, and you couldn't separate them:
- **S7** — fixed scale 50, alpha 0.10
- **S8** — fixed scale 50, alpha 0.20

(Optional, if GPUs are free: S3 config with alpha 0.10 / 0.20, to also see larger alpha under a learnable temperature.)

Total: 6 core arms (S1-S6) + 2 alpha arms (S7-S8) = 8 experiments (up to 10 with the optional alpha-under-learnable-temperature probes). All 1000 steps, same data/seed. No new baseline run is required — the existing `qwen3-0.6b-scratch-baseline` is the control for every arm (see the Logging section).

## Per-arm verdict to report

For each arm, report the trajectories of `log_logit_scale`, `logit_std`, `entropy`, `train/loss` (plus the other logged metrics) and a one-line call:
- learnable arms (S1-S4): did scale move off 2.64? did logit_std rise? loss stable?
- fixed arms (S5-S6): did entropy fall / logit_std stay healthy? loss stable?
- alpha arms (S7-S8): did larger alpha change entropy/logit_std behavior or destabilize loss? Watch `norm_ratio` — larger alpha means a larger contribution relative to token norm, which can shift behavior for reasons unrelated to selection quality (don't misattribute that to better selection).

Bring the trajectories back for review before promoting any arm to a full run.

