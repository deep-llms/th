"""EmbHub smoke run diagnostics callback (Test 3).

Logs 7 diagnostic metrics every N steps during training to detect:
- Selection not sharpening (entropy stays flat)
- Anchor collapse (pairwise cosine → 1)
- Dead anchors (routing concentrates on a few)

Add to Trainer via:
    from diagnostics.smoke_callback import EmbHubSmokeCallback
    trainer = Trainer(..., callbacks=[EmbHubSmokeCallback(model, tokenizer, log_every=50)])
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import TrainerCallback


METRIC_KEYS = [
    "embhub/entropy", "embhub/logit_std", "embhub/logit_scale",
    "embhub/top10_anchor_mass", "embhub/dead_anchor_frac",
    "embhub/anchor_pairwise_cos", "embhub/norm_ratio",
]


class EmbHubSmokeCallback(TrainerCallback):

    def __init__(self, model, tokenizer, log_every=50, probe_text=None, csv_path=None):
        self.model = model
        self.tokenizer = tokenizer
        self.log_every = log_every
        self.probe_text = probe_text or [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models require large amounts of data.",
            "Cross-lingual transfer enables models to work across languages.",
            "The global economy is influenced by many factors.",
        ]
        self.probe_ids = tokenizer(
            self.probe_text, return_tensors="pt", padding=True, truncation=True, max_length=128
        )["input_ids"]
        self.csv_path = csv_path
        self.csv_file = None
        self.csv_writer = None

    def _ensure_csv(self, output_dir):
        if self.csv_writer is not None:
            return
        path = self.csv_path or os.path.join(output_dir, "smoke_metrics.csv")
        file_exists = os.path.isfile(path)
        self.csv_file = open(path, "a", newline="")
        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=["step"] + METRIC_KEYS + ["train/loss"],
        )
        if not file_exists:
            self.csv_writer.writeheader()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every != 0 or state.global_step == 0:
            return
        if not hasattr(self.model, "embhub"):
            return

        hub = self.model.embhub
        embedding = self.model.get_input_embeddings()
        device = next(hub.parameters()).device
        probe = self.probe_ids.to(device)

        with torch.no_grad():
            token_emb = embedding(probe)
            diag = hub.compute_diagnostics(token_emb)

        metrics = {
            "embhub/entropy": diag["entropy_mean"],
            "embhub/logit_std": diag["logit_std"],
            "embhub/logit_scale": diag["logit_scale"],
            "embhub/top10_anchor_mass": diag["top10_anchor_mass"],
            "embhub/dead_anchor_frac": diag["dead_anchor_fraction"],
            "embhub/anchor_pairwise_cos": diag["anchor_pairwise_cosine"],
            "embhub/norm_ratio": diag["norm_ratio"],
        }

        # Save to CSV (main process only)
        if state.is_world_process_zero:
            self._ensure_csv(args.output_dir)
            row = {"step": state.global_step, **metrics}
            if state.log_history:
                last_log = state.log_history[-1]
                row["train/loss"] = last_log.get("loss", "")
            self.csv_writer.writerow(row)
            self.csv_file.flush()

        # Print summary periodically (main process only)
        if state.is_world_process_zero and state.global_step % (self.log_every * 10) == 0:
            print(f"  [EmbHub step {state.global_step}] "
                  f"entropy={diag['entropy_mean']:.3f} "
                  f"logit_std={diag['logit_std']:.3f} "
                  f"scale={diag['logit_scale']:.2f} "
                  f"dead={diag['dead_anchor_fraction']:.2%} "
                  f"anchor_cos={diag['anchor_pairwise_cosine']:.4f} "
                  f"norm_ratio={diag['norm_ratio']:.4f}")

        # Log to wandb if available
        if args.report_to and "wandb" in args.report_to:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(metrics, step=state.global_step)
            except ImportError:
                pass

    def on_train_end(self, args, state, control, **kwargs):
        if self.csv_file is not None:
            self.csv_file.close()
