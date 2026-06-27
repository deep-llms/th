"""Run all 10 smoke-test arms sequentially, each using all 8 GPUs.

Each arm runs smoke_train.py via accelerate launch with num_train_epochs=1.
The LR scheduler sees the full ~31.5K steps, identical to the real training run.
The script monitors training progress and terminates each arm after the target
step count, so the first 1000 steps have the exact same LR as the full run.

Usage:
  python run_smoke_tests.py
  python run_smoke_tests.py --arms S1 S3 S6
  python run_smoke_tests.py --stop-at-step 2000
"""

import argparse
import csv
import json
import logging
import os
import signal
import subprocess
import sys
import time

logger = logging.getLogger(__name__)

ARM_CONFIGS = {
    "S1":  {"scale_no_wd": True,  "scale_lr_mult": 1,  "scale_init": 14, "fixed_scale": False, "alpha": 0.05},
    "S2":  {"scale_no_wd": False, "scale_lr_mult": 75, "scale_init": 14, "fixed_scale": False, "alpha": 0.05},
    "S3":  {"scale_no_wd": True,  "scale_lr_mult": 75, "scale_init": 14, "fixed_scale": False, "alpha": 0.05},
    "S4":  {"scale_no_wd": True,  "scale_lr_mult": 75, "scale_init": 30, "fixed_scale": False, "alpha": 0.05},
    "S5":  {"scale_no_wd": True,  "scale_lr_mult": 1,  "scale_init": 30, "fixed_scale": True,  "alpha": 0.05},
    "S6":  {"scale_no_wd": True,  "scale_lr_mult": 1,  "scale_init": 50, "fixed_scale": True,  "alpha": 0.05},
    "S7":  {"scale_no_wd": True,  "scale_lr_mult": 1,  "scale_init": 50, "fixed_scale": True,  "alpha": 0.10},
    "S8":  {"scale_no_wd": True,  "scale_lr_mult": 1,  "scale_init": 50, "fixed_scale": True,  "alpha": 0.20},
    "S9":  {"scale_no_wd": True,  "scale_lr_mult": 75, "scale_init": 14, "fixed_scale": False, "alpha": 0.10},
    "S10": {"scale_no_wd": True,  "scale_lr_mult": 75, "scale_init": 14, "fixed_scale": False, "alpha": 0.20},
    # S3 alpha variants (same as S3 but with different alpha)
    "S3_a01":  {"scale_no_wd": True, "scale_lr_mult": 75, "scale_init": 14, "fixed_scale": False, "alpha": 0.10},
    "S3_a02":  {"scale_no_wd": True, "scale_lr_mult": 75, "scale_init": 14, "fixed_scale": False, "alpha": 0.20},
    "S3_a015": {"scale_no_wd": True, "scale_lr_mult": 75, "scale_init": 14, "fixed_scale": False, "alpha": 0.15},
    "S3_a025": {"scale_no_wd": True, "scale_lr_mult": 75, "scale_init": 14, "fixed_scale": False, "alpha": 0.25},
    "S3_a03":  {"scale_no_wd": True, "scale_lr_mult": 75, "scale_init": 14, "fixed_scale": False, "alpha": 0.30},
    "S3_a05":  {"scale_no_wd": True, "scale_lr_mult": 75, "scale_init": 14, "fixed_scale": False, "alpha": 0.50},
    "S3_a10":  {"scale_no_wd": True, "scale_lr_mult": 75, "scale_init": 14, "fixed_scale": False, "alpha": 1.00},
}

ALL_ARMS = list(ARM_CONFIGS.keys())
DEFAULT_OUTPUT_BASE = "/opt/dlami/nvme/smoke_test_outputs"


def build_cmd(arm_name, arm_cfg, output_dir, data_dir, stop_at_step, save_token_ids=False):
    cmd = [
        "accelerate", "launch", "smoke_train.py",
        "--config_name", "Qwen/Qwen3-0.6B",
        "--tokenizer_name", "Qwen/Qwen3-0.6B",
        "--data_dir", data_dir,
        "--block_size", "2048",
        "--preprocessing_num_workers", "160",
        "--num_hub_embeddings", "1000",
        "--alpha", str(arm_cfg["alpha"]),
        "--scale_init", str(arm_cfg["scale_init"]),
        "--scale_lr_mult", str(arm_cfg["scale_lr_mult"]),
        "--output_dir", output_dir,
        "--seed", "42",
        "--bf16",
        "--ddp_timeout", "21600",
        "--per_device_train_batch_size", "16",
        "--gradient_accumulation_steps", "4",
        "--num_train_epochs", "1",
        "--learning_rate", "3e-4",
        "--lr_scheduler_type", "cosine_with_min_lr",
        "--lr_scheduler_kwargs", '{"min_lr_rate": 0.1}',
        "--warmup_steps", "500",
        "--weight_decay", "0.1",
        "--adam_beta1", "0.9",
        "--adam_beta2", "0.95",
        "--max_grad_norm", "1.0",
        "--logging_steps", "10",
        "--save_steps", "250",
        "--dataloader_num_workers", "8",
        "--run_name", f"smoke-{arm_name}",
        "--report_to", "wandb",
    ]

    if arm_cfg["scale_no_wd"]:
        cmd.append("--scale_no_wd")
    if arm_cfg["fixed_scale"]:
        cmd.append("--fixed_scale")
    if save_token_ids:
        cmd += [
            "--save_token_ids",
            "--token_ids_output_dir", os.path.join(output_dir, "token_ids"),
        ]

    return cmd


def wait_for_step(proc, smoke_csv_path, target_step, poll_interval=10):
    """Poll the smoke_metrics.csv until target_step is reached, then terminate."""
    while proc.poll() is None:
        time.sleep(poll_interval)

        if not os.path.isfile(smoke_csv_path):
            continue

        try:
            with open(smoke_csv_path) as f:
                rows = list(csv.DictReader(f))
            if rows:
                last_step = int(rows[-1].get("step", 0))
                if last_step >= target_step:
                    logger.info(f"    Reached step {last_step} >= {target_step}, waiting 120s for checkpoint save...")
                    time.sleep(120)
                    logger.info(f"    Sending SIGTERM...")
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    proc.wait(timeout=60)
                    return last_step
        except (ValueError, OSError):
            continue

    return -1


def main():
    parser = argparse.ArgumentParser(description="Run smoke tests sequentially")
    parser.add_argument("--arms", nargs="+", default=ALL_ARMS, choices=ALL_ARMS)
    parser.add_argument("--output-base", default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--data-dir", default="/opt/dlami/nvme/embhub_data/Qwen_Qwen3-0.6B/train")
    parser.add_argument("--stop-at-step", type=int, default=1000)
    parser.add_argument("--save-token-ids", action="store_true",
                        help="Also dump training token IDs (for frequent-word analysis; decode offline)")
    parser.add_argument("--log", default="smoke_tests.log")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.log, mode="w"),
        ],
    )

    os.makedirs(args.output_base, exist_ok=True)

    completed = []
    start_time = time.time()

    logger.info(f"Running {len(args.arms)} smoke arms sequentially (8 GPUs each)")
    logger.info(f"Arms: {args.arms}")
    logger.info(f"Stop at step: {args.stop_at_step}")
    logger.info("")

    for arm_name in args.arms:
        arm_cfg = ARM_CONFIGS[arm_name]
        output_dir = os.path.join(args.output_base, arm_name)
        log_path = os.path.join(args.output_base, f"{arm_name}.log")
        smoke_csv_path = os.path.join(output_dir, "smoke_metrics.csv")

        cmd = build_cmd(arm_name, arm_cfg, output_dir, args.data_dir, args.stop_at_step,
                        save_token_ids=args.save_token_ids)

        fixed_str = "fixed" if arm_cfg["fixed_scale"] else "learnable"
        logger.info(f"  START: {arm_name} ({fixed_str}, alpha={arm_cfg['alpha']}, "
                     f"init={arm_cfg['scale_init']}, wd={'on' if not arm_cfg['scale_no_wd'] else 'off'}, "
                     f"lr_mult={arm_cfg['scale_lr_mult']})")

        arm_start = time.time()

        env = os.environ.copy()
        env["NCCL_NVLS_ENABLE"] = "0"
        env["WANDB_PROJECT"] = "cross_lingual_embedding_hub"
        env["WANDB_MODE"] = "offline"

        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(
                cmd, stdout=log_file, stderr=subprocess.STDOUT,
                preexec_fn=os.setsid, env=env,
            )
            last_step = wait_for_step(proc, smoke_csv_path, args.stop_at_step)

        arm_elapsed = time.time() - arm_start

        if last_step >= args.stop_at_step:
            status = f"STOPPED at step {last_step}"
        elif proc.returncode == 0:
            status = "OK (finished epoch)"
        else:
            status = f"FAILED (code {proc.returncode})"

        logger.info(f"  DONE:  {arm_name} — {status}  [{arm_elapsed:.0f}s]")
        if proc.returncode and proc.returncode not in (0, -signal.SIGTERM):
            logger.info(f"         See log: {log_path}")

        completed.append({
            "arm": arm_name, "config": arm_cfg, "returncode": proc.returncode,
            "output_dir": output_dir, "log_path": log_path, "elapsed": arm_elapsed,
            "last_step": last_step,
        })

        if arm_name != args.arms[-1]:
            logger.info(f"    Sleeping 30s before next arm...")
            time.sleep(30)

    total_elapsed = time.time() - start_time
    logger.info(f"\nAll {len(completed)} arms done in {total_elapsed:.0f}s")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SMOKE TEST SUMMARY")
    logger.info("=" * 70)

    for job in completed:
        arm = job["arm"]
        cfg = job["config"]
        fixed_str = "fixed" if cfg["fixed_scale"] else "learnable"

        csv_path = os.path.join(job["output_dir"], "smoke_metrics.csv")
        final_line = ""
        if os.path.isfile(csv_path):
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            if rows:
                last = rows[-1]
                final_line = (f"step={last.get('step', '?')} "
                              f"scale={float(last.get('embhub/logit_scale', 0)):.2f} "
                              f"logit_std={float(last.get('embhub/logit_std', 0)):.4f} "
                              f"entropy={float(last.get('embhub/entropy', 0)):.4f}")

        logger.info(f"\n  {arm} ({fixed_str}, alpha={cfg['alpha']}, init={cfg['scale_init']}, "
                     f"wd={'on' if not cfg['scale_no_wd'] else 'off'}, lr_mult={cfg['scale_lr_mult']}) "
                     f"[{job['elapsed']:.0f}s]:")
        if final_line:
            logger.info(f"    {final_line}")
        else:
            logger.info(f"    No metrics (see {job['log_path']})")

    logger.info("")


if __name__ == "__main__":
    main()
