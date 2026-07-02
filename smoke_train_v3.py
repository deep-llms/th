"""Train a causal LM with EmbHub V3/V2-concat/V2-topk variants.

Adapted from smoke_train.py. Supports all new hub types and placements.

Data loading, tokenization, and training loop are identical to smoke_train.py.
The only differences are the hub injection and save/load paths.
"""

import json
import logging
import math
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

from model_wrapper_v3 import inject_embhub_v3, save_embhub_v3

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str | None = field(default=None)
    config_name: str | None = field(default=None)
    tokenizer_name: str | None = field(default=None)
    cache_dir: str | None = field(default=None)
    token: str | None = field(default=None)
    trust_remote_code: bool = field(default=False)


@dataclass
class DataArguments:
    data_dir: str = field(default="data/sampled")
    block_size: int | None = field(default=None)
    preprocessing_num_workers: int | None = field(default=None)
    overwrite_cache: bool = field(default=False)


@dataclass
class EmbHubV3Arguments:
    no_embhub: bool = field(default=False, metadata={"help": "Train without EmbHub (baseline)"})
    freeze_base: bool = field(default=False, metadata={"help": "Freeze base model, train only hub"})
    hub_type: str = field(default="v3", metadata={"help": "Hub type: v3, v2_concat, v2_topk"})
    num_hub_embeddings: int = field(default=1000, metadata={"help": "Number of anchor embeddings"})
    placement: str = field(default="embedding", metadata={"help": "Placement: embedding or mid"})
    layer_idx: int = field(default=10, metadata={"help": "Layer index for mid-layer placement"})
    # V3-specific
    num_heads: int = field(default=1, metadata={"help": "Number of heads (V4)"})
    gate_bias_init: float = field(default=-5.0, metadata={"help": "Gate bias init for V3"})
    # V2-concat-specific
    use_mlp: bool = field(default=False, metadata={"help": "Use MLP before concat (V2b)"})
    # V2-topk-specific
    top_k: int = field(default=10, metadata={"help": "Number of top-k anchors (V2c/V6)"})
    weighting: str = field(default="raw_softmax", metadata={"help": "Weighting: raw_softmax, renormalized, none"})
    tail_mode: str = field(default="none", metadata={"help": "Tail mode: none, tail, buckets"})
    num_buckets: int = field(default=10, metadata={"help": "Number of buckets for V2c+buckets"})
    # V6/V6f-specific
    r_budget: float = field(default=0.3, metadata={"help": "V6f residual norm budget (fraction of concept norm)"})
    p_only: float = field(default=0.10, metadata={"help": "V6 target probability for concept-only mode"})
    p_both: float = field(default=0.40, metadata={"help": "V6 target probability for combined mode"})
    anneal_steps: int = field(default=2000, metadata={"help": "V6 curriculum anneal steps"})
    # Temperature
    scale_init: float = field(default=14.0, metadata={"help": "Initial logit scale"})
    scale_lr_mult: float = field(default=75.0, metadata={"help": "LR multiplier for log_logit_scale"})
    scale_no_wd: bool = field(default=True, metadata={"help": "Disable weight decay on log_logit_scale"})


@dataclass
class FreqArguments:
    save_token_ids: bool = field(default=False)
    token_ids_output_dir: str = field(default="temp/frequent_words_by_step")


class TokenIdWriter:
    def __init__(self, output_dir, rank):
        self.output_dir = output_dir
        self.rank = rank
        self.total_seqs = 0
        os.makedirs(output_dir, exist_ok=True)
        self.f = open(os.path.join(output_dir, f"ids_rank{rank}.jsonl"), "w", encoding="utf-8")

    def write_batch(self, input_ids):
        for ids in input_ids.tolist():
            self.total_seqs += 1
            self.f.write(json.dumps(ids) + "\n")

    def flush(self):
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


class EmbHubV3Trainer(Trainer):

    def __init__(self, embhub_args=None, id_writer=None, **kwargs):
        self._embhub_args = embhub_args
        self._id_writer = id_writer
        super().__init__(**kwargs)

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self._id_writer is not None:
            self._id_writer.write_batch(inputs["input_ids"])
        return super().training_step(model, inputs, num_items_in_batch)

    def create_optimizer(self):
        if self._embhub_args is None or self._embhub_args.no_embhub:
            return super().create_optimizer()
        if self._embhub_args.scale_lr_mult == 1.0 and not self._embhub_args.scale_no_wd:
            return super().create_optimizer()

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


class SaveEmbHubV3Callback(TrainerCallback):
    def __init__(self, model_args, data_args, embhub_args):
        self.model_args = model_args
        self.data_args = data_args
        self.embhub_args = embhub_args

    def on_save(self, args, state, control, **kwargs):
        if not args.should_save:
            return
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not self.embhub_args.no_embhub:
            save_embhub_v3(kwargs["model"], checkpoint_dir)
        save_train_config(checkpoint_dir, self.model_args, self.data_args, self.embhub_args, args)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, EmbHubV3Arguments, TrainingArguments, FreqArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, embhub_args, training_args, freq_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, embhub_args, training_args, freq_args = parser.parse_args_into_dataclasses()

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

    set_seed(training_args.seed)

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Checkpoint detected: {last_checkpoint}. Resuming training.")

    config_kwargs = {"cache_dir": model_args.cache_dir, "token": model_args.token, "trust_remote_code": model_args.trust_remote_code}
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

    tokenizer_name = model_args.tokenizer_name or model_args.model_name_or_path
    if tokenizer_name is None:
        raise ValueError("Must set --tokenizer_name when training from scratch")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=model_args.cache_dir, token=model_args.token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, config=config,
            cache_dir=model_args.cache_dir, token=model_args.token, trust_remote_code=model_args.trust_remote_code,
        )
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params / 2**20:.2f}M params")

    # Inject hub
    hub = None
    if not embhub_args.no_embhub:
        hub = inject_embhub_v3(
            model,
            hub_type=embhub_args.hub_type,
            num_embeddings=embhub_args.num_hub_embeddings,
            num_heads=embhub_args.num_heads,
            gate_bias_init=embhub_args.gate_bias_init,
            use_mlp=embhub_args.use_mlp,
            top_k=embhub_args.top_k,
            weighting=embhub_args.weighting,
            tail_mode=embhub_args.tail_mode,
            num_buckets=embhub_args.num_buckets,
            r_budget=embhub_args.r_budget,
            p_only=embhub_args.p_only,
            p_both=embhub_args.p_both,
            anneal_steps=embhub_args.anneal_steps,
            placement=embhub_args.placement,
            layer_idx=embhub_args.layer_idx,
            freeze_base=embhub_args.freeze_base,
        )
        with torch.no_grad():
            hub.log_logit_scale.fill_(math.log(embhub_args.scale_init))

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total params: {total_params:,}, Trainable: {trainable_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)")
    if hub is not None:
        logger.info(f"Hub: type={embhub_args.hub_type}, placement={embhub_args.placement}, "
                     f"layer_idx={embhub_args.layer_idx}, num_embeddings={embhub_args.num_hub_embeddings}, "
                     f"scale={hub.log_logit_scale.exp().item():.1f}, "
                     f"scale_lr_mult={embhub_args.scale_lr_mult}")

    # Load data (shard-aware)
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

    def tokenize_function(examples):
        return tokenizer(examples["text"], add_special_tokens=False)

    with training_args.main_process_first(desc="tokenization"):
        tokenized_dataset = raw_dataset.map(
            tokenize_function, batched=True, num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names, load_from_cache_file=not data_args.overwrite_cache,
        )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if hasattr(config, "max_position_embeddings"):
            max_pos = config.max_position_embeddings
        else:
            max_pos = 1024
        if block_size > max_pos:
            block_size = min(1024, max_pos) if max_pos > 0 else 1024
    else:
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    def group_texts(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples}
        total_length = (len(concatenated[list(examples.keys())[0]]) // block_size) * block_size
        result = {k: [t[i:i+block_size] for i in range(0, total_length, block_size)]
                  for k, t in concatenated.items()}
        result["labels"] = result["input_ids"].copy()
        return result

    with training_args.main_process_first(desc="grouping"):
        lm_dataset = tokenized_dataset.map(
            group_texts, batched=True, num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    train_dataset = lm_dataset.shuffle(seed=training_args.seed)
    logger.info(f"Training: {train_dataset.num_rows:,} sequences of {block_size} tokens")

    # Callbacks
    callbacks = [SaveEmbHubV3Callback(model_args, data_args, embhub_args)]
    if not embhub_args.no_embhub:
        from diagnostics.smoke_callback_v3 import EmbHubV3SmokeCallback
        callbacks.append(EmbHubV3SmokeCallback(model, tokenizer, log_every=25))

        if hasattr(hub, "current_step"):
            class V6StepCallback(TrainerCallback):
                def on_step_end(self, args, state, control, **kwargs):
                    model.embhub.current_step.fill_(state.global_step)
            callbacks.append(V6StepCallback())

    id_writer = None
    if freq_args.save_token_ids:
        rank = training_args.local_process_index
        id_writer = TokenIdWriter(freq_args.token_ids_output_dir, rank)
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

        class TokenIdFlushCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                id_writer.flush()
        callbacks.append(TokenIdFlushCallback())

    trainer = EmbHubV3Trainer(
        embhub_args=embhub_args,
        id_writer=id_writer,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        callbacks=callbacks,
    )

    logger.info("*** Train ***")
    checkpoint = training_args.resume_from_checkpoint or last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    if id_writer is not None:
        id_writer.close()

    trainer.save_model()
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if training_args.should_save:
        if not embhub_args.no_embhub:
            save_embhub_v3(model, training_args.output_dir)
        save_train_config(training_args.output_dir, model_args, data_args, embhub_args, training_args)
    logger.info(f"Training complete. Saved to: {training_args.output_dir}")


if __name__ == "__main__":
    main()
