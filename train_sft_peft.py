import os
import sys
import torch
from dataclasses import dataclass, field
from typing import List, Union, Optional

from transformers import HfArgumentParser, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from trl import SFTConfig, SFTTrainer
from peft import PeftModel, LoraConfig, TaskType, get_peft_model

from PIL import Image as PILImage
import json

from utils import set_random_seed
from preprocess_dataset import preprocess_dataset
from preprocess_dataset_train import preprocess_dataset_train

from utils import load_model_and_processor

import datetime

def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"{datetime.datetime.now():%Y%m%d_%H%M%S}_train.log")
    log_file = open(log_path, "w", encoding="utf-8")
    class Tee:
        def __init__(self, *streams):
            self._streams = streams
        def write(self, data):
            for s in self._streams:
                s.write(data)
                s.flush()
            return len(data)
        def flush(self):
            for s in self._streams:
                s.flush()
        def isatty(self):
            return getattr(self._streams[0], "isatty", lambda: False)()
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    return log_file


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="llava-hf/llava-1.5-7b-hf", metadata={"help": "Path to pretrained model or model identifier."})
    trust_remote_code: bool = field(default=False, metadata={"help": "Trust remote code."})
    lora_enable: bool = field(default=True, metadata={"help": "Enable LoRA."})
    # freeze_vision: bool = field(default=True, metadata={"help": "Freeze vision backbone."}) # Not used????
    # Trainable layers
    finetune_vision_layers: bool = field(default=False, metadata={"help": "Finetune vision layers."})
    finetune_language_layers: bool = field(default=True, metadata={"help": "Finetune language layers."})
    finetune_attention_modules: bool = field(default=True, metadata={"help": "Finetune attention modules."})
    finetune_mlp_modules: bool = field(default=True, metadata={"help": "Finetune MLP modules."})
    # Target modules for customization
    target_layers: str = field(default=None, metadata={"help": "Target modules. For LoRA, the default is 'all-linear'."})
    # Lora config
    lora_r: int = field(default=16, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.00, metadata={"help": "LoRA dropout."})
    lora_bias: str = field(default="none", metadata={"help": "LoRA bias."})
    use_rslora: bool = field(default=False, metadata={"help": "Use RSLoRA."})
    loftq_config: str = field(default=None, metadata={"help": "LoFTQ config."})
    random_state: int = field(default=3407, metadata={"help": "Random state for LoRA."})

@dataclass
class DataArguments:
    dataset_name: List[str] = field(
        default_factory=list,
        metadata={"help": "One or more datasets."}
    )
    # Optional parallel list of sample counts (same length as dataset_name)
    dataset_sample_num: Optional[List[int]] = field(
        default=None,
        metadata={"help": "Per-dataset sample counts; -1 = all. Must match number of datasets if provided."}
    )
    max_train_samples: int = field(default=None, metadata={"help": "Maximum number of training samples to use."})
    is_disjoint_comb:  bool = field(default=False, metadata={"help": "Whether multiple datasets are disjoint subsets."})


@dataclass
class TrainingArguments(SFTConfig):
    output_dir: str = field(default="./models/SFT", metadata={"help": "Output directory for model predictions and checkpoints."})
    bf16: bool = field(default=True, metadata={"help": "Use bf16 precision."})
    resume_from_checkpoint: str = field(default=None, metadata={"help": "Path to a checkpoint to resume training from."})
    warmup_ratio: float = field(default=0.03, metadata={"help": "Warmup ratio for learning rate scheduler."})
    max_seq_length: int = field(default=2048, metadata={"help": "Maximum sequence length for the model."})
    gradient_checkpointing: str = field(default="unsloth", metadata={"help": "Use gradient checkpointing."})
    # LoRA checkpoint to resume from
    # is_resume_lora: bool = field(default=False, metadata={"help": "Whether to resume from a LoRA checkpoint."})
    overwrite: bool = field(default=False, metadata={"help": "Whether to overwrite the output directory."})


    # Deepspeed
    num_train_epochs: int = field(default=1, metadata={"help": "Number of training epochs."})
    dataset_num_proc: int = field(default=4, metadata={"help": "Number of processes to use for dataset processing."})
    # dataloader_num_workers: int = field(default=32, metadata={"help": "Number of workers for data loading."})
    per_device_train_batch_size: int = field(default=4, metadata={"help": "Batch size per device during training."})
    gradient_accumulation_steps: int = field(default=4, metadata={"help": "Number of gradient accumulation steps."})

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
    save_steps: int = field(default=0.1, metadata={"help": "Save steps."})  # If save_steps is smaller than 1, will be interpreted as ratio of total training steps.

    # Seed
    seed: int = field(default=42, metadata={"help": "Random seed."})

    # Unslothのデモにあったやつ
    remove_unused_columns: bool = field(default=False, metadata={"help": "Remove unused columns in the dataset."})
    dataset_text_field: str = field(default="", metadata={"help": "Text field in the dataset."})
    dataset_kwargs: dict = field(default_factory=lambda: {"skip_prepare_dataset": True}, metadata={"help": "Additional arguments for the dataset."})
    max_length: int = field(default=2048, metadata={"help": "Maximum length of the input sequences."})


def main():
    print("[PEFT] Starting SFT training...")
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # training_args.output_dir = f"{training_args.output_dir}/{os.path.basename(data_args.dataset_name)}_{os.path.basename(model_args.model_name_or_path)}"

    # If already exists, skip
    if os.path.exists(os.path.join(training_args.output_dir, "tokenizer.model")) or os.path.exists(os.path.join(training_args.output_dir, "tokenizer.json")):
        if not training_args.overwrite:
            print(f"==> Output directory already exists and is not empty. Skipping training.")
            print(f"====> {training_args.output_dir} ")
            print("==" * 20)
            sys.exit(0)
        else:
            print("Overwriting directory:")
            print(f"====> {training_args.output_dir} ")
            pass
    

    # --- Prepare args ---
    if model_args.target_layers is not None:
        model_args.target_layers = model_args.target_layers.split(",")
    if training_args.gradient_checkpointing.lower() in ["t", "true"]:
        training_args.gradient_checkpointing = True
    elif training_args.gradient_checkpointing.lower() in ["f", "false"]:
        training_args.gradient_checkpointing = False

    # save args to output_dir as json
    os.makedirs(training_args.output_dir, exist_ok=True)

    log_file = setup_logger(training_args.output_dir)

    from dataclasses import asdict

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
        with open(os.path.join(training_args.output_dir, filename), 'w') as f:
            json.dump(d, f, indent=4)
    

    args_to_json(model_args, "model_args.json")
    args_to_json(data_args, "data_args.json")
    args_to_json(training_args, "training_args.json")

    # Set seed
    print(f"Setting random seed: {training_args.seed}")
    set_random_seed(training_args.seed)
    
    # Set PEFT
    if model_args.lora_enable:
        print("==> Starting LoRA training...")
        # Load model and processor
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
            # loftq_config=model_args.loftq_config,
            task_type=TaskType.CAUSAL_LM,
            target_modules="all-linear"
        )
        model = get_peft_model(model, peft_config)

        if model_args.target_layers is not None:
            print(f"==> Target layers: {model_args.target_layers}")
            # Enable grads for the target layers
            for name, param in model.named_parameters():
                if "lora" in name:
                    if not any(target in name for target in model_args.target_layers):
                        param.requires_grad = False
                        param.data.zero_() # Zero out the parameters not used.
                        print(f"  - {name} is frozen, and zeroed out")
            print("Set manual trainable parameters for the target layers.")
    else:
        raise NotImplementedError("Only LoRA training is implemented.")
        # Load model and processor
        print(f"Loading model (full): {model_args.model_name_or_path}")
        model, processor = FastVisionModel.from_pretrained(
            model_name=model_args.model_name_or_path,
            full_finetuning=True,     
            use_gradient_checkpointing=training_args.gradient_checkpointing,
        )

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # enable grads for the target modules
        for name, param in model.named_parameters():
            if any(target in name for target in model_args.target_layers):
                param.requires_grad = True
                print(f"  - {name} is trainable")
        print("Set manual trainable parameters for the target modules.")


    # --- Load dataset ---
    def load_dataset_with_check(dataset_name, max_train_samples, split="train"):
        print(f"Loading dataset: {dataset_name}")
        if os.path.isdir(dataset_name):
            ds = load_from_disk(dataset_name)
        else:
            ds = load_dataset(dataset_name, split=split, trust_remote_code=True)

        if max_train_samples is not None and max_train_samples > 0:
            ds = ds.shuffle(seed=training_args.seed).select(range(max_train_samples))

        # Qwenの場合は画像のサイズが一定値以上でないとエラーになるので、サイズをチェックする
        if "qwen" in model_args.model_name_or_path.lower():
            print("Qwen model detected, checking image sizes...")
            min_size = 28

            def is_large_enough(example):
                if not isinstance(example["image"], PILImage.Image):
                    img = PILImage.open(example["image"]).convert("RGB")
                else:
                    img = example["image"]
                w, h = img.size
                return w >= min_size and h >= min_size

            original_len = len(ds)
            ds = ds.filter(is_large_enough)
            print(f"Removed {original_len - len(ds)} examples with image size < {min_size}px")
        return ds

    # Preprocess dataset
    if isinstance(data_args.dataset_name, list) and len(data_args.dataset_name) > 1:
        print("Multiple datasets detected, combining datasets...")
        if data_args.is_disjoint_comb:
            print("Datasets are sampled from disjoint subsets.")
            indices_list = []
            total_samples = sum([num if num is not None and num > 0 else 0 for num in data_args.dataset_sample_num])
            print(f"Total samples across all datasets: {total_samples}")
            shuffled_indices = list(range(total_samples))
            import random
            random.seed(training_args.seed)
            random.shuffle(shuffled_indices)
            start_idx = 0
            for ds_name, ds_num in zip(data_args.dataset_name, data_args.dataset_sample_num):
                num_samples = ds_num if ds_num is not None and ds_num > 0 else 0
                ds_indices = shuffled_indices[start_idx:start_idx + num_samples]
                indices_list.append(ds_indices)
                print(f"- Assigned {num_samples} samples to dataset {ds_name}: {start_idx} to {start_idx + num_samples - 1}")
                start_idx += num_samples

            processed_datasets = []
            for ds_name, ds_num in zip(data_args.dataset_name, data_args.dataset_sample_num):
                ds = load_dataset_with_check(ds_name, ds_num, split="train", indices=indices_list.pop(0))
                split_name = ds_name.split("/")[-2]
                subset_name = ds_name.split("/")[-1]
                processed_ds = preprocess_dataset_train(ds, split_name=split_name, subset_name=subset_name, model_name=model_args.model_name_or_path, peft_ver=False)
                processed_datasets.append(processed_ds)
                print(f"--- {ds_name}: {len(processed_ds)}")
            # Combine datasets
            from itertools import chain
            processed_train_dataset = list(chain.from_iterable(processed_datasets))
            print(f"Combined dataset size: {len(processed_train_dataset)}")
        else:
            print("Datasets are sampled from overlapping subsets.")
            processed_datasets = []
            for ds_name, ds_num in zip(data_args.dataset_name, data_args.dataset_sample_num):
                ds = load_dataset_with_check(ds_name, ds_num, split="train")
                # try:
                #     processed_ds = preprocess_dataset(ds, dataset_id=ds_name, model_name=model_args.model_name_or_path, text_only=True)
                # except:
                split_name = ds_name.split("/")[-2]
                subset_name = ds_name.split("/")[-1]
                processed_ds = preprocess_dataset_train(ds, split_name=split_name, subset_name=subset_name, model_name=model_args.model_name_or_path, peft_ver=True)

                processed_datasets.append(processed_ds)
                print(f"--- {ds_name}: {len(processed_ds)}")
        # Combine datasets
        from itertools import chain
        processed_train_dataset = list(chain.from_iterable(processed_datasets))
        print(f"Combined dataset size: {len(processed_train_dataset)}")
    else:
        if isinstance(data_args.dataset_name, list):
            data_args.dataset_name = data_args.dataset_name[0]
            data_args.dataset_sample_num = data_args.dataset_sample_num[0] if data_args.dataset_sample_num is not None else -1
        ds = load_dataset_with_check(data_args.dataset_name, data_args.dataset_sample_num, split="train")
        
        split_name = data_args.dataset_name.split("/")[-2]
        subset_name = data_args.dataset_name.split("/")[-1]
        processed_train_dataset = preprocess_dataset_train(ds, split_name=split_name, subset_name=subset_name, model_name=model_args.model_name_or_path, peft_ver=True)

    # Training: https://github.com/huggingface/peft/blob/main/examples/sft/train.py
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

    # Save LoRA model
    print("Saving LoRA model...")
    if not os.path.exists(os.path.join(training_args.output_dir, "lora")):
        os.makedirs(os.path.join(training_args.output_dir, "lora"))
    trainer.model.save_pretrained(os.path.join(training_args.output_dir, "lora"))
    processor.save_pretrained(os.path.join(training_args.output_dir, "lora"))
    print(f"LoRA model saved to {os.path.join(training_args.output_dir, 'lora')}")

    # Save merged model
    new_dir = os.path.join(training_args.output_dir)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    # model = model.merge_and_unload()
    # model.save_pretrained(new_dir)
    # processor.save_pretrained(new_dir)
    # print(f"Merged model saved to {new_dir}")

    # save lora model
    model.save_pretrained(new_dir, processor)
    processor.save_pretrained(new_dir)
    print(f"LoRA saved to {new_dir}")

    log_file.close()


if __name__ == "__main__":
    main()
