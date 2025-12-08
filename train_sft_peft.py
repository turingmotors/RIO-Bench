import os
import sys
import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import json
import torch
from datasets import load_dataset, load_from_disk
from PIL import Image as PILImage
from transformers import HfArgumentParser
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, TaskType, get_peft_model

from utils import set_random_seed, Tee, load_model_and_processor
from preprocess_dataset_train import preprocess_dataset_train


def setup_logger(output_dir: str) -> "IO[str]":
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(
        output_dir,
        f"{datetime.datetime.now():%Y%m%d_%H%M%S}_train.log",
    )
    log_file = open(log_path, "w", encoding="utf-8")

    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    return log_file


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="llava-hf/llava-1.5-7b-hf",
        metadata={"help": "Path to pretrained model or model identifier."},
    )
    trust_remote_code: bool = field(default=False, metadata={"help": "Trust remote code."})
    lora_enable: bool = field(default=True, metadata={"help": "Enable LoRA."})
    # Trainable layers
    finetune_vision_layers: bool = field(default=False, metadata={"help": "Finetune vision layers."})
    finetune_language_layers: bool = field(default=True, metadata={"help": "Finetune language layers."})
    finetune_attention_modules: bool = field(default=True, metadata={"help": "Finetune attention modules."})
    finetune_mlp_modules: bool = field(default=True, metadata={"help": "Finetune MLP modules."})
    # Target modules for customization
    target_layers: str = field(
        default=None,
        metadata={"help": "Target modules. For LoRA, the default is 'all-linear'."},
    )
    # LoRA config
    lora_r: int = field(default=16, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "LoRA dropout."})
    lora_bias: str = field(default="none", metadata={"help": "LoRA bias."})
    use_rslora: bool = field(default=False, metadata={"help": "Use RSLoRA."})
    loftq_config: str = field(default=None, metadata={"help": "LoFTQ config."})
    random_state: int = field(default=3407, metadata={"help": "Random state for LoRA."})


@dataclass
class DataArguments:
    dataset_name: List[str] = field(
        default_factory=list,
        metadata={"help": "One or more datasets."},
    )
    # Optional parallel list of sample counts (same length as dataset_name)
    dataset_sample_num: Optional[List[int]] = field(
        default=None,
        metadata={"help": "Per-dataset sample counts; -1 = all. Must match number of datasets if provided."},
    )
    max_train_samples: int = field(
        default=None,
        metadata={"help": "Maximum number of training samples to use."},
    )
    is_disjoint_comb: bool = field(
        default=False,
        metadata={"help": "Whether multiple datasets are disjoint subsets."},
    )


@dataclass
class TrainingArguments(SFTConfig):
    output_dir: str = field(
        default="./models/SFT",
        metadata={"help": "Output directory for model predictions and checkpoints."},
    )
    bf16: bool = field(default=True, metadata={"help": "Use bf16 precision."})
    resume_from_checkpoint: str = field(
        default=None,
        metadata={"help": "Path to a checkpoint to resume training from."},
    )
    warmup_ratio: float = field(default=0.03, metadata={"help": "Warmup ratio for learning rate scheduler."})
    max_seq_length: int = field(default=2048, metadata={"help": "Maximum sequence length for the model."})
    gradient_checkpointing: str = field(default="unsloth", metadata={"help": "Use gradient checkpointing."})
    overwrite: bool = field(default=False, metadata={"help": "Whether to overwrite the output directory."})

    # Training setup
    num_train_epochs: int = field(default=1, metadata={"help": "Number of training epochs."})
    dataset_num_proc: int = field(default=4, metadata={"help": "Number of processes to use for dataset processing."})
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per device during training."},
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "Number of gradient accumulation steps."},
    )

    # Optimizer
    learning_rate: float = field(default=2e-5, metadata={"help": "Learning rate for the optimizer."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for the optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "Learning rate scheduler type."})

    # Logging
    logging_steps: int = field(default=10, metadata={"help": "Logging steps."})
    report_to: str = field(default="wandb", metadata={"help": "Reporting tool."})

    # Save
    save_strategy: str = field(default="steps", metadata={"help": "Save strategy."})
    save_steps: float = field(
        default=0.1,
        metadata={"help": "Save steps or ratio of total training steps (if < 1)."},
    )

    # Seed
    seed: int = field(default=42, metadata={"help": "Random seed."})

    # From Unsloth demo: keep dataset as-is and provide pre-tokenized inputs
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Remove unused columns in the dataset."},
    )
    dataset_text_field: str = field(default="", metadata={"help": "Text field in the dataset."})
    dataset_kwargs: dict = field(
        default_factory=lambda: {"skip_prepare_dataset": True},
        metadata={"help": "Additional arguments for the dataset."},
    )
    max_length: int = field(default=2048, metadata={"help": "Maximum length of the input sequences."})


def main():
    print("[PEFT] Starting SFT training...")
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # If already exists, skip
    if (
        os.path.exists(os.path.join(training_args.output_dir, "tokenizer.model"))
        or os.path.exists(os.path.join(training_args.output_dir, "tokenizer.json"))
    ):
        if not training_args.overwrite:
            print("==> Output directory already exists and is not empty. Skipping training.")
            print(f"====> {training_args.output_dir} ")
            print("==" * 20)
            sys.exit(0)
        else:
            print("Overwriting directory:")
            print(f"====> {training_args.output_dir} ")

    # --- Prepare args ---
    if model_args.target_layers is not None:
        model_args.target_layers = model_args.target_layers.split(",")
    if training_args.gradient_checkpointing.lower() in ["t", "true"]:
        training_args.gradient_checkpointing = True
    elif training_args.gradient_checkpointing.lower() in ["f", "false"]:
        training_args.gradient_checkpointing = False

    # Save args to output_dir as json
    os.makedirs(training_args.output_dir, exist_ok=True)
    log_file = setup_logger(training_args.output_dir)

    def filter_json_serializable(d):
        def is_json_serializable(v):
            try:
                json.dumps(v)
                return True
            except Exception:
                return False

        return {k: v for k, v in d.items() if is_json_serializable(v)}

    def args_to_json(args, filename):
        d = asdict(args)
        d = filter_json_serializable(d)
        with open(os.path.join(training_args.output_dir, filename), "w") as f:
            json.dump(d, f, indent=4)

    args_to_json(model_args, "model_args.json")
    args_to_json(data_args, "data_args.json")
    args_to_json(training_args, "training_args.json")

    # Set seed
    print(f"Setting random seed: {training_args.seed}")
    set_random_seed(training_args.seed)

    # --- Load model & configure PEFT ---
    if model_args.lora_enable:
        print("==> Starting LoRA training...")
        print(f"Loading model: {model_args.model_name_or_path}")
        model, processor = load_model_and_processor(model_args.model_name_or_path)
        print(model)

        assert model_args.loftq_config is None, "LoFTQ is not implemented yet."

        print("Configuring PEFT model...")
        print("Trained Params: ")
        peft_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            bias=model_args.lora_bias,
            use_rslora=model_args.use_rslora,
            task_type=TaskType.CAUSAL_LM,
            target_modules="all-linear",
        )
        model = get_peft_model(model, peft_config)

        if model_args.target_layers is not None:
            print(f"==> Target layers: {model_args.target_layers}")
            # Enable grads only for the target layers
            for name, param in model.named_parameters():
                if "lora" in name and not any(target in name for target in model_args.target_layers):
                    param.requires_grad = False
                    param.data.zero_()
                    print(f"  - {name} is frozen, and zeroed out")
            print("Set manual trainable parameters for the target layers.")
    else:
        raise NotImplementedError("Only LoRA training is implemented for this script.")

    # --- Load dataset ---
    def load_dataset_with_check(
        dataset_name: str,
        max_train_samples: Optional[int],
        split: str = "train",
        seed: int = training_args.seed,
        indices: Optional[List[int]] = None,
    ):
        print(f"Loading dataset: {dataset_name}")
        if os.path.isdir(dataset_name):
            ds = load_from_disk(dataset_name)
        else:
            ds = load_dataset(dataset_name, split=split, trust_remote_code=True)

        if max_train_samples is not None and max_train_samples > 0:
            if indices is not None:
                print(f"Selecting {max_train_samples} samples with provided indices...")
                ds = ds.select(indices)
            else:
                print(f"Selecting {max_train_samples} random samples from the dataset... seed={seed}")
                ds = ds.shuffle(seed=seed).select(range(max_train_samples))

        # Special handling for Qwen: enforce minimum image size
        if "qwen" in model_args.model_name_or_path.lower():
            print("Qwen model detected, checking image sizes...")
            min_size = 28

            def is_large_enough(example):
                img = example["image"]
                if not isinstance(img, PILImage.Image):
                    img = PILImage.open(img).convert("RGB")
                w, h = img.size
                return w >= min_size and h >= min_size

            original_len = len(ds)
            ds = ds.filter(is_large_enough)
            print(f"Removed {original_len - len(ds)} examples with image size < {min_size}px")
        return ds

    # --- Preprocess dataset ---
    def preprocess_one_dataset(ds_name: str, ds, model_args, peft_ver: bool):
        split_name = ds_name.split("/")[-2]
        subset_name = ds_name.split("/")[-1]
        return preprocess_dataset_train(
            ds,
            split_name=split_name,
            subset_name=subset_name,
            model_name=model_args.model_name_or_path,
            peft_ver=peft_ver,
        )

    if isinstance(data_args.dataset_name, list) and len(data_args.dataset_name) > 1:
        print("Multiple datasets detected, combining datasets...")
        from itertools import chain
        import random

        if data_args.is_disjoint_comb:
            print("Datasets are sampled from disjoint subsets.")
            indices_list = []
            total_samples = sum(
                [num if num is not None and num > 0 else 0 for num in data_args.dataset_sample_num]
            )
            print(f"Total samples across all datasets: {total_samples}")
            shuffled_indices = list(range(total_samples))
            random.seed(training_args.seed)
            random.shuffle(shuffled_indices)

            start_idx = 0
            for ds_name, ds_num in zip(data_args.dataset_name, data_args.dataset_sample_num):
                num_samples = ds_num if ds_num is not None and ds_num > 0 else 0
                ds_indices = shuffled_indices[start_idx : start_idx + num_samples]
                indices_list.append(ds_indices)
                print(
                    f"- Assigned {num_samples} samples to dataset {ds_name}: "
                    f"{start_idx} to {start_idx + num_samples - 1}"
                )
                start_idx += num_samples

            processed_datasets = []
            for ds_name, ds_num in zip(data_args.dataset_name, data_args.dataset_sample_num):
                ds = load_dataset_with_check(
                    ds_name,
                    ds_num,
                    split="train",
                    indices=indices_list.pop(0),
                )
                # disjoint case: original code used peft_ver=False
                processed_ds = preprocess_one_dataset(ds_name, ds, model_args, peft_ver=False)
                processed_datasets.append(processed_ds)
                print(f"--- {ds_name}: {len(processed_ds)}")
            processed_train_dataset = list(chain.from_iterable(processed_datasets))
            print(f"Combined dataset size: {len(processed_train_dataset)}")
        else:
            print("Datasets are sampled from overlapping subsets.")
            processed_datasets = []
            for ds_name, ds_num in zip(data_args.dataset_name, data_args.dataset_sample_num):
                ds = load_dataset_with_check(ds_name, ds_num, split="train")
                # overlapping case: original code used peft_ver=True
                processed_ds = preprocess_one_dataset(ds_name, ds, model_args, peft_ver=True)
                processed_datasets.append(processed_ds)
                print(f"--- {ds_name}: {len(processed_ds)}")
            processed_train_dataset = list(chain.from_iterable(processed_datasets))
            print(f"Combined dataset size: {len(processed_train_dataset)}")
    else:
        # Single dataset case
        if isinstance(data_args.dataset_name, list):
            data_args.dataset_name = data_args.dataset_name[0]
            data_args.dataset_sample_num = (
                data_args.dataset_sample_num[0] if data_args.dataset_sample_num is not None else -1
            )
        ds = load_dataset_with_check(
            data_args.dataset_name,
            data_args.dataset_sample_num,
            split="train",
        )
        # single dataset: original code used peft_ver=True
        processed_train_dataset = preprocess_one_dataset(
            data_args.dataset_name, ds, model_args, peft_ver=True
        )

    # --- Training ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_train_dataset,
        peft_config=peft_config,
        processing_class=processor,
    )
    trainer.accelerator.print(f"{trainer.model}")
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Save LoRA model (adapter) separately
    print("Saving LoRA model...")
    lora_dir = os.path.join(training_args.output_dir, "lora")
    os.makedirs(lora_dir, exist_ok=True)
    trainer.model.save_pretrained(lora_dir)
    processor.save_pretrained(lora_dir)
    print(f"LoRA model saved to {lora_dir}")

    # Save full PEFT weights + processor to output_dir
    new_dir = training_args.output_dir
    os.makedirs(new_dir, exist_ok=True)
    model.save_pretrained(new_dir, processor)
    processor.save_pretrained(new_dir)
    print(f"LoRA saved to {new_dir}")

    log_file.close()


if __name__ == "__main__":
    main()
