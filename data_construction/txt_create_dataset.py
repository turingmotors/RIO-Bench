# Create Txt-Clean and Txt-Attack datasets by combining original TextVQA
# 1. Append a suffix to each question to encourage concise answers
# 2. Convert all images to RGB JPEG format in memory
# 3. Save the processed datasets to disk, including txt-attack metadata

import os
import json
import glob
import argparse
from typing import Dict, List, Set, Tuple, Optional
from datasets import load_dataset, Dataset, DatasetDict
import random
import pickle
from PIL import Image
from datasets import Image as HFImage
from tqdm import tqdm

import io
from PIL import Image as PILImage

import copy


SUFFIX = " Answer the question using a single word or phrase."

# Keys in meta that overlap with dataset entry semantics and should be removed
META_DROP_KEYS = {
    "image_id",
    "question_id",
    "answer",
    "attack_word",
}


parser = argparse.ArgumentParser(description="Create RIO-Bench txt datasets.")
parser.add_argument(
    "--splits",
    nargs="+",
    choices=["validation", "train"],
    default=["validation", "train"],
    help="TextVQA splits to process.",
)
parser.add_argument(
    "--attack-levels",
    nargs="+",
    choices=["misleading", "correct"],
    default=["misleading"],
    help="Text attack levels to build.",
)
parser.add_argument("--overwrite", action="store_true", help="Overwrite existing datasets.")
parser.add_argument("--skip-clean", action="store_true", help="Skip creating txt_clean dataset.")
args = parser.parse_args()


def convert_img(img: Image.Image, quality: int = 95) -> Image.Image:
    """Convert image to RGB JPEG format in memory."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf)


def load_as_jpeg(path: str, quality: int = 95) -> Image.Image:
    """Load an image and convert it to RGB JPEG (handles PNG, etc.)."""
    with Image.open(path) as img:
        return convert_img(img, quality=quality)


# -------------------------------------------
# Process both validation and train splits
# -------------------------------------------
for split, split_short, method_list, method_map in [
    (
        "validation",
        "val",
        ["txt_bucket_val_far", "txt_bucket_val_mid"],
        {
            "txt_bucket_val_far": "far",
            "txt_bucket_val_mid": "mid",
        },
    ),
    (
        "train",
        "train",
        ["txt_bucket_train_far", "txt_bucket_train_mid"],
        {
            "txt_bucket_train_far": "far",
            "txt_bucket_train_mid": "mid",
        },
    ),
]:
    if split not in set(args.splits):
        continue
    # Load original TextVQA split
    ds = load_dataset("facebook/textvqa", split=split)
    print(f"Loaded TextVQA {split} set with {len(ds)} entries.")

    # -------------------------------------------
    # Save original split with suffix (txt_clean)
    # -------------------------------------------
    if not args.skip_clean:
        entries = []
        for i, item in enumerate(ds):
            if i % 1000 == 0:
                print(f"  Processing {i}/{len(ds)} (clean)")
            # Work on a copy to avoid mutating the original dataset
            entry = dict(item)

            # Add suffix to question (exactly once)
            question = entry["question"]
            entry["question"] = question + SUFFIX

            # Convert image to RGB JPEG format
            entry["image"] = convert_img(entry["image"])
            entries.append(entry)

        dataset_clean = Dataset.from_list(entries)
        clean_out_path = f"../data/RIO-Bench/hf_dataset/{split_short}/txt_clean/original_with_suffix"
        os.makedirs(os.path.dirname(clean_out_path), exist_ok=True)
        if os.path.exists(clean_out_path) and args.overwrite:
            import shutil
            shutil.rmtree(clean_out_path)
        if not os.path.exists(clean_out_path):
            dataset_clean.save_to_disk(clean_out_path)
            print(f"Saved original TextVQA {split} set with suffix to {clean_out_path}")
        else:
            print(f"Dataset {clean_out_path} already exists, skipping...")

    # -------------------------------------------
    # Txt-Attack datasets (far/mid) with meta
    # -------------------------------------------
    DATASET_SAVE_DIR = f"../data/RIO-Bench/hf_dataset/{split_short}/txt_attack"
    IMAGE_DIR = f"../data/{split_short}/txt_attack/"

    for method in method_list:
        for attack_level in args.attack_levels:
            # --- Load meta.json for this method & split ---
            if attack_level == "misleading":
                method_image_dir = os.path.join(IMAGE_DIR, method)
                meta_json_path = os.path.join(
                    method_image_dir,
                    f"text_attack_{method}_meta.json",
                )
                out_suffix = method_map[method]
            elif attack_level == "correct":
                method_image_dir = os.path.join(IMAGE_DIR, method, "correct")
                meta_json_path = os.path.join(
                    method_image_dir,
                    f"text_attack_{method}_correct_meta.json",
                )
                out_suffix = f"correct_{method_map[method]}"
            else:
                raise ValueError(f"Unknown attack level: {attack_level}")

            if not os.path.exists(meta_json_path):
                raise FileNotFoundError(f"Meta file not found: {meta_json_path}")
            with open(meta_json_path, "r") as f:
                raw_meta = json.load(f)

            # Clean meta to avoid overlapping keys with entry
            meta_dict: Dict[int, dict] = {}
            for qid_str, meta in raw_meta.items():
                cleaned_meta = {k: v for k, v in meta.items() if k not in META_DROP_KEYS}
                # keys in raw_meta are strings like "34602" -> cast to int
                meta_dict[int(qid_str)] = cleaned_meta
            print(f"Loaded {len(meta_dict)} cleaned meta entries from {meta_json_path}")

            # --- Open-ended questions dataset with attacked images ---
            print(f"Creating MCP: {method} ({attack_level}) dataset...")
            entries = []
            for i, item in tqdm(enumerate(ds)):
                image_id = item["image_id"]
                question_id = item["question_id"]

                image_path = glob.glob(
                    os.path.join(method_image_dir, "images", f"{image_id}_{question_id}*")
                )[0]
                image = load_as_jpeg(image_path)
                if isinstance(image, dict) and "bytes" in image:
                    print(
                        f"Warning: question_id {question_id} image is in bytes dict format, "
                        "converting to PIL Image."
                    )
                    exit()

                # Deep copy to mirror original behavior
                entry = copy.deepcopy(item)
                entry["image"] = image

                # Append suffix to question (exactly once)
                question = entry["question"]
                entry["question"] = question + SUFFIX

                # Attach meta (without overlapping keys)
                meta = meta_dict.get(question_id)
                entry["meta"] = meta

                entries.append(entry)

                if i == 0:
                    print(entry)

            print(f"{method} ({attack_level}): Created {len(entries)} entries.")
            dataset_attack = Dataset.from_list(entries)
            out_path = f"{DATASET_SAVE_DIR}/open_ended_{out_suffix}"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            if os.path.exists(out_path) and args.overwrite:
                import shutil
                shutil.rmtree(out_path)
            if not os.path.exists(out_path):
                dataset_attack.save_to_disk(out_path)
                print(f"Saved {method} ({attack_level}) dataset to {out_path}")
            else:
                print(f"Dataset {out_path} already exists, skipping...")
