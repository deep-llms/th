# EmbHub Diagnostics Test Results

```
======================================================================
  Test 1 — Wiring Sanity (alpha=0 must match base model)
======================================================================
  Max absolute difference                      0.000000 ✓ PASS
  Mean absolute difference                     0.000000

======================================================================
  Test 2 — Static Initialization Check
======================================================================
  --- New formula (cosine + learnable temperature) ---
  Logit scale (exp(log_logit_scale))          14.000001
  Logit std                                    0.439044 ✓ PASS  (expected: 0.3-0.6)
  Entropy mean                                 6.811187 ✓ PASS  (expected: 6.70-6.88)
  Entropy std                                  0.004055
  Uniform entropy (log 1000)                   6.907755
  Effective anchor count                     907.947711
  Mean max weight                              0.003831 ✓ PASS  (expected: 0.003-0.006)
  Norm ratio (||alpha*contrib|| / ||tok||)     0.001879
  Anchor pairwise cosine                      -0.000349

  --- Old formula (dot-product / sqrt(d)) for comparison ---
  Old logit std                                0.000397  (confirms uniform — the old bug)
  Old entropy mean                             6.907755  (= log(1000), fully uniform)
  Old effective anchors                      999.999619  (all equal weight)
  Old max weight mean                          0.001001  (= 1/1000, uniform)

======================================================================
  Test 4 — Gradient Symmetry Contrast
======================================================================
  --- New (cosine+temp) ---
  Grad pairwise cosine                         0.931417  (< 1.0, anchors differentiate)
  Grad norm CV                                 0.094755  (> 0, varied gradient magnitudes)
  --- Old (dot/sqrt(d)) ---
  Grad pairwise cosine                         1.000000  (= 1.0, all identical — clone trap)
  Grad norm CV                                 0.000076  (= 0, no differentiation)

======================================================================
  Summary
======================================================================
  Test 1 (Wiring sanity):        PASS
  Test 2 (Static init):          PASS
  Test 4 (Gradient symmetry):    PASS

  ✓ All tests passed. Proceed to smoke training run (Test 3).
```

## Key takeaways

- The old dot-product/sqrt(d) formula is confirmed dead — logit std 0.0004, entropy = uniform, gradients perfectly identical across all anchors
- The new cosine + learnable temp formula is alive — logit std 0.44, entropy 6.81 (below uniform 6.91), gradients differentiated (cosine 0.93 vs 1.00)
- All numbers match the predicted ranges from the test plan exactly
- Test 3 (smoke run) callback is implemented and auto-attached to training — will log the 7 dynamic metrics to wandb every 50 steps

## What was implemented

1. **Updated `hub_layer_v2.py`** — new cosine + learnable temperature formula:
   - `F.normalize` on both token embeddings and anchors for selection
   - `log_logit_scale` learnable parameter (init = log(14) ~ 2.639)
   - Raw (unnormalized) anchors used as values
   - `compute_diagnostics()` method for all 7 metrics

2. **`diagnostics/embhub_diagnostics.py`** — Tests 1, 2, 4:
   - Test 1: alpha=0 wiring sanity
   - Test 2: static initialization check with old vs new comparison
   - Test 4: gradient symmetry contrast

3. **`diagnostics/smoke_callback.py`** — Training callback for Test 3:
   - Logs 7 metrics every 50 steps to wandb
   - Prints summary every 500 steps

4. **Updated `train.py`** — auto-adds smoke callback when training with EmbHub
