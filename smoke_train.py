"""Train a causal LM (optionally with EmbHub) on multilingual data.

Adapted from HuggingFace's run_clm.py example:
https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py

Data is loaded from per-language directories saved by prepare_data.py.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from itertools import chain

import datasets
import torch
from datasets import load_from_disk, concatenate_datasets

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from model_wrapper_v2 import inject_embhub, load_model_with_embhub, save_embhub

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Model checkpoint for weights initialization. Don't set if training from scratch."},
    )
    config_name: str | None = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name_or_path"},
    )
    tokenizer_name: str | None = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name_or_path"},
    )
    cache_dir: str | None = field(
        default=None,
        metadata={"help": "Where to store pretrained models downloaded from huggingface.co"},
    )
    token: str | None = field(
        default=None,
        metadata={"help": "HF auth token for downloading gated models/tokenizers"},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust remote code from the Hub"},
    )


@dataclass
class DataArguments:
    data_dir: str = field(
        default="data/sampled",
        metadata={"help": "Directory containing per-language raw text datasets (saved by prepare_data.py)"},
    )
    block_size: int | None = field(
        default=None,
        metadata={"help": "Optional input sequence length after tokenization. Defaults to model max length."},
    )
    preprocessing_num_workers: int | None = field(
        default=None,
        metadata={"help": "Number of processes for preprocessing"},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets"},
    )


@dataclass
class EmbHubArguments:
    num_hub_embeddings: int = field(default=1000, metadata={"help": "Number of hub embeddings"})
    alpha: float = field(default=0.05, metadata={"help": "Hub scaling factor"})
    freeze_base: bool = field(default=False, metadata={"help": "Freeze base model, train only hub"})
    no_embhub: bool = field(default=False, metadata={"help": "Train original model without EmbHub layer"})
    scale_init: float = field(default=14.0, metadata={"help": "Initial logit scale (exp of log_logit_scale)"})
    fixed_scale: bool = field(default=False, metadata={"help": "Fix logit scale (not learnable)"})
    scale_lr_mult: float = field(default=1.0, metadata={"help": "LR multiplier for log_logit_scale"})
    scale_no_wd: bool = field(default=False, metadata={"help": "Disable weight decay on log_logit_scale"})


@dataclass
class FreqArguments:
    """Optional: dump training token IDs during training (for frequent-word analysis).

    Saves raw token IDs (NOT decoded text) of EVERY training sequence, in training
    order, one per rank. Decoding is done offline by count_words_by_lang.py, which
    slices "up to step S" via the formula S * batch * grad_accum (recorded in
    meta.json). No markers / pre-specified steps needed. Keeps the training hot
    path fast. Default off; does not affect normal training runs at all."""
    save_token_ids: bool = field(default=False,
                                 metadata={"help": "Write token IDs of each training sequence to disk"})
    token_ids_output_dir: str = field(default="temp/frequent_words_by_step",
                                      metadata={"help": "Where to write token IDs + meta.json"})


class TokenIdWriter:
    """Writes the raw token IDs of EVERY training sequence (per rank) as JSONL,
    one JSON int-list per line, in training order. File: ids_rank{N}.jsonl.

    No markers: the offline reader maps line -> step via S * seqs_per_step
    (seqs_per_step = per_device_batch * grad_accum, recorded in meta.json).
    Writing IDs (not decoded text) keeps the training hot path fast."""

    def __init__(self, output_dir, rank):
        self.output_dir = output_dir
        self.rank = rank
        self.total_seqs = 0
        os.makedirs(output_dir, exist_ok=True)
        self.f = open(os.path.join(output_dir, f"ids_rank{rank}.jsonl"), "w", encoding="utf-8")
        logger.info(f"Rank {rank}: writing token IDs to {output_dir}/ids_rank{rank}.jsonl")

    def write_batch(self, input_ids):
        # input_ids: (batch, block_size) tensor for ONE micro-batch.
        # training_step is called once per micro-batch (grad_accum times per
        # optimizer step), so this writes batch rows each call, in training order.
        for ids in input_ids.tolist():
            self.total_seqs += 1
            self.f.write(json.dumps(ids) + "\n")

    def flush(self):
        # Called every optimizer step so completed steps survive a SIGTERM.
        self.f.flush()

    def close(self):
        self.f.close()


def save_train_config(save_dir, model_args, data_args, embhub_args, training_args):
    config = {
        "model": asdict(model_args),
        "data": asdict(data_args),
        "embhub": asdict(embhub_args),
        "training": {
            k: v for k, v in training_args.to_dict().items()
            if v is not None and v != "" and k not in ("_n_gpu", "local_rank")
        },
    }
    with open(os.path.join(save_dir, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)


class EmbHubTrainer(Trainer):
    """Trainer with custom optimizer groups for log_logit_scale.

    If id_writer is provided, training_step also writes the token IDs of
    each sequence before running the (unchanged) real forward+backward."""

    def __init__(self, embhub_args=None, id_writer=None, **kwargs):
        self._embhub_args = embhub_args
        self._id_writer = id_writer
        super().__init__(**kwargs)

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Optional token-ID dump (no-op for normal runs where writer is None).
        # inputs["input_ids"] is the exact data this step trains on.
        if self._id_writer is not None:
            self._id_writer.write_batch(inputs["input_ids"])
        return super().training_step(model, inputs, num_items_in_batch)

    def create_optimizer(self):
        if self._embhub_args is None or self._embhub_args.no_embhub:
            return super().create_optimizer()
        if self._embhub_args.scale_lr_mult == 1.0 and not self._embhub_args.scale_no_wd:
            return super().create_optimizer()

        # Custom groups: separate log_logit_scale from default Trainer logic
        decay_names = set(self.get_decay_parameter_names(self.model))

        scale_params = []
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "log_logit_scale" in name:
                scale_params.append(param)
            elif name in decay_names:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        base_lr = self.args.learning_rate
        base_wd = self.args.weight_decay

        groups = [
            {"params": decay_params, "weight_decay": base_wd, "lr": base_lr},
            {"params": no_decay_params, "weight_decay": 0.0, "lr": base_lr},
        ]

        if scale_params:
            groups.append({
                "params": scale_params,
                "weight_decay": 0.0 if self._embhub_args.scale_no_wd else base_wd,
                "lr": base_lr * self._embhub_args.scale_lr_mult,
            })

        self.optimizer = torch.optim.AdamW(
            groups,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        return self.optimizer


class SaveEmbHubCallback(TrainerCallback):
    def __init__(self, model_args, data_args, embhub_args):
        self.model_args = model_args
        self.data_args = data_args
        self.embhub_args = embhub_args

    def on_save(self, args, state, control, **kwargs):
        if not args.should_save:
            return
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not self.embhub_args.no_embhub:
            save_embhub(kwargs["model"], checkpoint_dir)
        save_train_config(checkpoint_dir, self.model_args, self.data_args, self.embhub_args, args)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, EmbHubArguments, TrainingArguments, FreqArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, embhub_args, training_args, freq_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, embhub_args, training_args, freq_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_process_index}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, distributed training: {training_args.parallel_mode.value == 'distributed'}, "
        f"16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Detect last checkpoint for resume
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Checkpoint detected: {last_checkpoint}. Resuming training.")

    # Load config
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        if model_args.config_name.endswith(".json") and os.path.isfile(model_args.config_name):
            with open(model_args.config_name) as f:
                config_dict = json.load(f)
            config = AutoConfig.for_model(**config_dict)
        else:
            config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        raise ValueError("Must set --model_name_or_path or --config_name")

    # Load tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    tokenizer_name = model_args.tokenizer_name or model_args.model_name_or_path
    if tokenizer_name is None:
        raise ValueError("Must set --tokenizer_name when training from scratch without --model_name_or_path")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    if model_args.model_name_or_path:
        if embhub_args.no_embhub:
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
            )
            logger.info(f"Loaded pretrained model: {model_args.model_name_or_path} (no EmbHub)")
        else:
            model, hub = load_model_with_embhub(
                model_args.model_name_or_path,
                num_embeddings=embhub_args.num_hub_embeddings,
                alpha=embhub_args.alpha,
                freeze_base=embhub_args.freeze_base,
                config=config,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
            )
            embhub_path = os.path.join(model_args.model_name_or_path, "embhub.pt")
            if os.path.isfile(embhub_path):
                logger.info(f"Loaded model with existing hub weights from: {model_args.model_name_or_path}")
            else:
                logger.info(f"Loaded model with fresh hub: {model_args.model_name_or_path}")
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params / 2**20:.2f}M params")

        if not embhub_args.no_embhub:
            hub = inject_embhub(
                model,
                num_embeddings=embhub_args.num_hub_embeddings,
                alpha=embhub_args.alpha,
                freeze_base=embhub_args.freeze_base,
            )

    # Apply scale settings
    if not embhub_args.no_embhub:
        import math
        with torch.no_grad():
            hub.log_logit_scale.fill_(math.log(embhub_args.scale_init))
        if embhub_args.fixed_scale:
            hub.log_logit_scale.requires_grad = False

    # Log model info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total params: {total_params:,}, Trainable: {trainable_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)")
    if not embhub_args.no_embhub:
        fixed_str = "FIXED" if embhub_args.fixed_scale else "learnable"
        logger.info(f"Hub embeddings: {hub.hub_embeddings.shape}, alpha: {embhub_args.alpha}, "
                     f"freeze_base: {embhub_args.freeze_base}, "
                     f"logit_scale: {hub.log_logit_scale.exp().item():.1f} ({fixed_str}), "
                     f"scale_lr_mult: {embhub_args.scale_lr_mult}, "
                     f"scale_no_wd: {embhub_args.scale_no_wd}")

    # Load per-language datasets (supports both single-dir and sharded layouts)
    datasets_list = []
    for lang_dir in sorted(os.listdir(data_args.data_dir)):
        lang_path = os.path.join(data_args.data_dir, lang_dir)
        if not os.path.isdir(lang_path):
            continue
        shard_dirs = sorted(
            os.path.join(lang_path, d) for d in os.listdir(lang_path)
            if d.startswith("shard_") and os.path.isdir(os.path.join(lang_path, d))
        )
        if shard_dirs:
            total = 0
            for sd in shard_dirs:
                ds = load_from_disk(sd)
                total += ds.num_rows
                datasets_list.append(ds)
            logger.info(f"[{lang_dir}] {total:,} documents ({len(shard_dirs)} shards)")
        else:
            ds = load_from_disk(lang_path)
            logger.info(f"[{lang_dir}] {ds.num_rows:,} documents")
            datasets_list.append(ds)

    if not datasets_list:
        raise ValueError(f"No datasets found in {data_args.data_dir}")

    raw_dataset = concatenate_datasets(datasets_list)
    logger.info(f"Combined: {raw_dataset.num_rows:,} documents")
    column_names = raw_dataset.column_names

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"], add_special_tokens=False)

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    # Determine block_size
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if hasattr(config, "max_position_embeddings"):
            max_pos = config.max_position_embeddings
        else:
            max_pos = 1024
        if block_size > max_pos:
            logger.warning(
                f"Tokenizer model_max_length ({block_size}) > max_position_embeddings ({max_pos}). "
                f"Using block_size={min(1024, max_pos)}."
            )
            block_size = min(1024, max_pos) if max_pos > 0 else 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"block_size ({data_args.block_size}) > tokenizer model_max_length ({tokenizer.model_max_length}). "
                f"Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Group texts into chunks of block_size
    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    with training_args.main_process_first(desc="grouping texts together"):
        lm_dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    train_dataset = lm_dataset.shuffle(seed=training_args.seed)
    logger.info(f"Training dataset: {train_dataset.num_rows:,} sequences of {block_size} tokens")

    # Initialize Trainer
    callbacks = [SaveEmbHubCallback(model_args, data_args, embhub_args)]
    if not embhub_args.no_embhub:
        from diagnostics.smoke_callback import EmbHubSmokeCallback
        callbacks.append(EmbHubSmokeCallback(model, tokenizer, log_every=25))

    # Optional: dump training token IDs (default off — normal runs unaffected).
    # Writes raw IDs of every sequence (fast); decode happens offline.
    id_writer = None
    if freq_args.save_token_ids:
        # MUST use local_process_index (the value the startup log prints as
        # "Process rank: N", correctly 0..7). Do NOT use training_args.local_rank:
        # under `accelerate launch` it is -1, so `... else 0` would make all 8
        # GPUs write to ids_rank0.jsonl (interleaved corruption).
        rank = training_args.local_process_index
        id_writer = TokenIdWriter(freq_args.token_ids_output_dir, rank)

        # Rank 0 writes meta.json: the config the offline reader needs to map
        # line index -> training step (lines per step = batch * grad_accum).
        if rank == 0:
            os.makedirs(freq_args.token_ids_output_dir, exist_ok=True)
            meta = {
                "tokenizer": tokenizer_name,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "world_size": training_args.world_size,
                "block_size": block_size,
                "seqs_per_step_per_rank": (training_args.per_device_train_batch_size
                                           * training_args.gradient_accumulation_steps),
            }
            with open(os.path.join(freq_args.token_ids_output_dir, "meta.json"), "w") as mf:
                json.dump(meta, mf, indent=2)
            logger.info(f"Wrote token-id meta.json: {meta}")

        # Flush each optimizer step so completed steps survive SIGTERM.
        class TokenIdFlushCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                id_writer.flush()
        callbacks.append(TokenIdFlushCallback())

    trainer = EmbHubTrainer(
        embhub_args=embhub_args,
        id_writer=id_writer,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        callbacks=callbacks,
    )

    # Training
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    if id_writer is not None:
        id_writer.close()

    trainer.save_model()

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Save EmbHub weights and train config (main process only)
    if training_args.should_save:
        if not embhub_args.no_embhub:
            save_embhub(model, training_args.output_dir)
        save_train_config(training_args.output_dir, model_args, data_args, embhub_args, training_args)
    logger.info(f"Training complete. Model saved to: {training_args.output_dir}")


if __name__ == "__main__":
    main()
