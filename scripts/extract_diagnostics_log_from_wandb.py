"""Extract EmbHub diagnostic metrics from wandb using the wandb API.

Usage:
  python scripts/extract_diagnostics_log_from_wandb.py
  python scripts/extract_diagnostics_log_from_wandb.py --run-path entity/project/run_id
  python scripts/extract_diagnostics_log_from_wandb.py --every 1000
"""

import argparse
import csv
import glob
import os

import wandb


EMBHUB_KEYS = [
    "embhub/entropy",
    "embhub/logit_std",
    "embhub/logit_scale",
    "embhub/top10_anchor_mass",
    "embhub/dead_anchor_frac",
    "embhub/anchor_pairwise_cos",
    "embhub/norm_ratio",
]

TRAIN_KEYS = [
    "train/loss",
    "train/learning_rate",
]


def find_latest_run_id(wandb_dir="wandb"):
    runs = sorted(glob.glob(os.path.join(wandb_dir, "run-*")))
    if not runs:
        return None
    dirname = os.path.basename(runs[-1])
    parts = dirname.split("-")
    return parts[-1] if len(parts) >= 3 else None


def main():
    parser = argparse.ArgumentParser(description="Extract EmbHub metrics from wandb")
    parser.add_argument("--run-path", default=None,
                        help="Wandb run path: entity/project/run_id (default: auto-detect from local wandb/)")
    parser.add_argument("--project", default="cross_lingual_embedding_hub", help="Wandb project name")
    parser.add_argument("--every", type=int, default=250, help="Extract every N steps (default: 250)")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: embhub_diagnostics_every{N}.csv)")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"embhub_diagnostics_every{args.every}.csv"

    api = wandb.Api()

    if args.run_path:
        run = api.run(args.run_path)
    else:
        run_id = find_latest_run_id()
        if run_id is None:
            print("No wandb runs found in wandb/. Use --run-path.")
            return
        entity = api.default_entity
        run_path = f"{entity}/{args.project}/{run_id}"
        print(f"Auto-detected run: {run_path}")
        run = api.run(run_path)

    print(f"Run: {run.name} ({run.id})")
    print(f"State: {run.state}")
    print(f"Extracting every {args.every} steps...")

    all_keys = ["_step"] + EMBHUB_KEYS + TRAIN_KEYS
    history = run.scan_history(keys=all_keys)

    rows = []
    for entry in history:
        step = entry.get("_step", 0)
        if step % args.every != 0 or step == 0:
            continue
        if not any(k in entry for k in EMBHUB_KEYS):
            continue
        row = {"step": step}
        for k in EMBHUB_KEYS + TRAIN_KEYS:
            if k in entry:
                row[k] = entry[k]
        rows.append(row)

    if not rows:
        print("No embhub metrics found. Training may not have reached the first extraction step yet.")
        return

    present_keys = ["step"] + [k for k in EMBHUB_KEYS + TRAIN_KEYS if any(k in r for r in rows)]

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=present_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Extracted {len(rows)} rows to {args.output}")
    print(f"Columns: {present_keys}")

    print(f"\nFirst 3 rows:")
    for r in rows[:3]:
        print(f"  step={r.get('step', '?'):>6}  entropy={r.get('embhub/entropy', ''):>8}  "
              f"scale={r.get('embhub/logit_scale', ''):>8}  "
              f"anchor_cos={r.get('embhub/anchor_pairwise_cos', ''):>8}")

    if len(rows) > 3:
        print(f"\nLast 3 rows:")
        for r in rows[-3:]:
            print(f"  step={r.get('step', '?'):>6}  entropy={r.get('embhub/entropy', ''):>8}  "
                  f"scale={r.get('embhub/logit_scale', ''):>8}  "
                  f"anchor_cos={r.get('embhub/anchor_pairwise_cos', ''):>8}")


if __name__ == "__main__":
    main()
