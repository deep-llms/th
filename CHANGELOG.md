# Changelog

## v2 (current) — Cosine + Learnable Temperature

### EmbHub Formula Change
- **Old (v1):** `softmax(token_emb @ anchors.T / sqrt(d))` — scaled dot-product
- **New (v2):** `softmax(normalize(token_emb) @ normalize(anchors).T * exp(log_logit_scale))` — cosine similarity with learnable temperature

**Why:** The v1 formula produced uniform attention at initialization (logit std ~0.0004, entropy = log(1000) = uniform). All anchors received identical gradients, so they remained clones and the hub layer was effectively dead. The v2 formula normalizes inputs for selection and uses a learnable temperature (init = 14, CLIP-style), giving logit std ~0.44 and differentiated gradients across anchors.

See `docs/embhub_diagnostics_test_plan_after_tried_version_1.md` for the full analysis and `diagnostics/embhub_diagnostics_test_plan_after_tried_version_1_test_results.md` for test results.

### Key changes
- `hub_layer_v2.py`: Cosine + learnable `log_logit_scale` parameter. Added `compute_diagnostics()` method.
- `train.py`: Rewritten based on HuggingFace's `run_clm.py`. Added `SaveEmbHubCallback` with `args.should_save` guard for multi-GPU. Added `EmbHubSmokeCallback` for diagnostic logging during training.
- `prepare_data.py`: Added eval set sampling (documents immediately after train cutoff, zero overlap). New folder structure: `data/{tokenizer}/train/{lang}` and `data/{tokenizer}/eval/{lang}`.
- `eval_perplexity.py`: Sliding-window perplexity following HuggingFace docs. Fixed token counting bug in HF reference code.
- `eval_benchmarks.py`: Cross-lingual benchmarks via lm-eval (XNLI, Belebele, XCOPA, XStoryCloze, PAWS-X, HellaSwag). Auto-patches lm-eval dataset paths for huggingface_hub 1.x compatibility.
- `diagnostics/embhub_diagnostics.py`: Tests 1, 2, 4 from the test plan.
- `diagnostics/smoke_callback.py`: Training callback logging 7 EmbHub metrics every 50 steps to wandb.

### Training config
- Data: 30B en + 5×300M other languages, 1 epoch (unique tokens, no repetition)
- Model: Qwen3-0.6B architecture, trained from scratch
- EmbHub: 1000 anchors, alpha=0.05, cosine + learnable temperature
- LR: 3e-4 cosine with min_lr 0.1, warmup 500 steps
- Batch: 16 × 4 × 8 GPUs = 512 sequences/step = ~1M tokens/step
- ~31,500 total steps

---

## v1 — Scaled Dot-Product (deprecated)

### EmbHub Formula
`softmax(token_emb @ anchors.T / sqrt(d))` with fixed scale `1/sqrt(d)`.

### Result
EmbHub was slightly worse than baseline at step 10,000 (~0.4 higher PPL). The training machine was lost before steps 20,000/30,000 could be evaluated. Post-hoc analysis revealed the formula produced uniform attention — the hub layer was born dead.

### Training config
- Data: 10B en + 5×100M other languages, 3 epochs (30B total, duplicated)
- Same model architecture and hyperparameters except warmup=750, epochs=3

### Files (not preserved, but can be reconstructed)
- `hub_layer_v2.py` used `self._scale = 1.0 / math.sqrt(embedding_dim)` and `softmax(token_emb @ anchors.T * self._scale)`
- No eval data, no diagnostics, no perplexity evaluation script
- No `args.should_save` guard in callbacks (multi-GPU race condition)
