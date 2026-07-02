"""EmbHub V3 smoke diagnostics callback.

Handles both embedding-layer and mid-layer placement. For mid-layer hubs,
captures the hidden state at the target layer to feed to compute_diagnostics.

Logs metrics every N steps to CSV + wandb. Adds gate_mean (V3) or w_mix_norm
(V2-concat/V2-topk) to the standard metrics.
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
    "embhub/gate_mean", "embhub/w_mix_norm", "embhub/w_anchor_norm",
    "embhub/topk_mass_total",
]


class EmbHubV3SmokeCallback(TrainerCallback):

    def __init__(self, model, tokenizer, log_every=25, probe_text=None, csv_path=None):
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
            fieldnames=["step"] + METRIC_KEYS,
        )
        if not file_exists:
            self.csv_writer.writeheader()

    def _get_diagnostics_input(self, probe):
        """Get the right input for compute_diagnostics based on placement."""
        hub = self.model.embhub
        placement = getattr(self.model, "_embhub_placement", "embedding")

        if placement == "embedding":
            embedding = self.model.get_input_embeddings()
            return embedding(probe)

        layer_idx = getattr(self.model, "_embhub_layer_idx", 0)
        from model_wrapper_v3 import _get_transformer_layers
        layers = _get_transformer_layers(self.model)
        target_layer = layers[layer_idx]

        captured = {}
        def capture_hook(mod, inp, out):
            if isinstance(out, tuple):
                captured["hidden"] = out[0].detach()
            else:
                captured["hidden"] = out.detach()

        handle = target_layer.register_forward_hook(capture_hook)
        self.model(probe)
        handle.remove()
        return captured["hidden"]

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every != 0 or state.global_step == 0:
            return
        if not hasattr(self.model, "embhub"):
            return

        hub = self.model.embhub
        device = next(hub.parameters()).device
        probe = self.probe_ids.to(device)

        with torch.no_grad():
            diag_input = self._get_diagnostics_input(probe)
            diag = hub.compute_diagnostics(diag_input)

        metrics = {
            "embhub/entropy": diag["entropy_mean"],
            "embhub/logit_std": diag["logit_std"],
            "embhub/logit_scale": diag["logit_scale"],
            "embhub/top10_anchor_mass": diag["top10_anchor_mass"],
            "embhub/dead_anchor_frac": diag["dead_anchor_fraction"],
            "embhub/anchor_pairwise_cos": diag["anchor_pairwise_cosine"],
            "embhub/norm_ratio": diag["norm_ratio"],
            "embhub/gate_mean": diag.get("gate_mean", ""),
            "embhub/w_mix_norm": diag.get("w_mix_norm", ""),
            "embhub/w_anchor_norm": diag.get("w_anchor_norm", ""),
            "embhub/topk_mass_total": diag.get("topk_mass_total", ""),
        }

        if state.is_world_process_zero:
            self._ensure_csv(args.output_dir)
            row = {"step": state.global_step, **metrics}
            self.csv_writer.writerow(row)
            self.csv_file.flush()

        if state.is_world_process_zero and state.global_step % (self.log_every * 10) == 0:
            placement = getattr(self.model, "_embhub_placement", "embedding")
            hub_type = getattr(self.model, "_embhub_hub_type", "?")
            extra = ""
            if "gate_mean" in diag:
                extra += f" gate={diag['gate_mean']:.4f}"
            if "w_mix_norm" in diag:
                extra += f" w_mix={diag['w_mix_norm']:.4f}"
            print(f"  [EmbHub-{hub_type} step {state.global_step} @{placement}] "
                  f"entropy={diag['entropy_mean']:.3f} "
                  f"scale={diag['logit_scale']:.2f} "
                  f"dead={diag['dead_anchor_fraction']:.2%} "
                  f"norm_ratio={diag['norm_ratio']:.4f}"
                  f"{extra}")

        if args.report_to and "wandb" in args.report_to:
            try:
                import wandb
                if wandb.run is not None:
                    clean = {k: v for k, v in metrics.items() if v != ""}
                    wandb.log(clean, step=state.global_step)
            except ImportError:
                pass

    def on_train_end(self, args, state, control, **kwargs):
        if self.csv_file is not None:
            self.csv_file.close()
