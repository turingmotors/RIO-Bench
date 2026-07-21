# Create Obj-Clean and Obj-Attack datasets by combining multiple-choice questions and adversarial examples
import argparse
import os
import glob
import json
from typing import Dict, List, Set, Tuple, Optional
from datasets import load_dataset, Dataset
import random
import pickle
from PIL import Image
from datasets import Image as HFImage
import numpy as np

import io
from PIL import Image as PILImage

from obj_create_multi_choice_data import prune_gt_to_pseudo_leaves


parser = argparse.ArgumentParser(description="Create RIO-Bench obj datasets.")
parser.add_argument("--overwrite", action="store_true", help="Overwrite existing datasets.")
parser.add_argument(
    "--splits",
    nargs="+",
    choices=["validation", "train"],
    default=["validation", "train"],
    help="TextVQA splits to process.",
)
parser.add_argument(
    "--levels",
    nargs="+",
    choices=["clean", "correct", "hard", "medium", "easy"],
    default=["clean", "hard", "medium", "easy"],
    help="Dataset levels to build.",
)
args = parser.parse_args()
OVERWRITE = args.overwrite


def convert_img(img: Image.Image, quality=95) -> Image.Image:
    """Convert image to RGB JPEG format in memory."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf)


def load_as_jpeg(path: str, quality: int = 95):
    """Load image and convert to RGB JPEG (handles PNG etc.)."""
    with Image.open(path) as img:
        return convert_img(img, quality=quality)


# Base directory for meta.json files
# Example: ../data/RIO-Bench/data/val/obj_attack/val_rc/easy/obj_attack_easy_meta.json
ASSETS_DIR = "assets/open_images"
TEXTVQA_META_DIR = "assets/textvqa_meta"
TEXTVQA_MCQ_DIR = "assets/mcq_data"

DATA_ROOT_DIR = "../data"
DATA_DIR = f"{DATA_ROOT_DIR}/RIO-Bench"
os.makedirs(DATA_DIR, exist_ok=True)

META_DROP_KEYS = {
    "image_id",
    "question_id",
    "answer",
    "choices",
    "attack_word",
}

# -------------------------------------------
# Process both train and validation splits
# -------------------------------------------
for split in args.splits:
    split_short = "val" if split == "validation" else "train"

    # Load original TextVQA split
    ds = load_dataset("facebook/textvqa", split=split)
    print(f"Loaded TextVQA {split} set with {len(ds)} entries.")

    # Load multiple-choice questions
    mcq_json_path = f"{TEXTVQA_MCQ_DIR}/textvqa_{split_short}_mcq_samples.json"
    with open(mcq_json_path, "r") as f:
        mcq_data = json.load(f)
    print(f"Loaded {len(mcq_data)} multiple-choice questions from {mcq_json_path}")
    mcq_data_dict = {item["question_id"]: item for item in mcq_data}

    config_name_list = ["rc"]  # e.g., "rc", "grid_white_28px", etc.
    config_name_map = {
        "rc": "random",
    }

    for config_name in config_name_list:
        print(f"Processing config: {config_name}")
        # ------------------------------------------
        # --- Multiple-choice questions dataset ---
        # ------------------------------------------
        QUESTION_FORMAT = (
            "Which object is present in the image? "
            "(A) {A} (B) {B} (C) {C} (D) {D}. "
            "Answer with only the option letter (A, B, C, or D)."
        )

        for level in args.levels:
            if level == "clean":
                mode = "obj_clean"
            else:
                mode = "obj_attack"

            DATASET_SAVE_DIR = f"{DATA_DIR}/hf_dataset/{split_short}/{mode}"
            os.makedirs(DATASET_SAVE_DIR, exist_ok=True)

            if level == "clean":
                out_path = f"{DATASET_SAVE_DIR}/mc_{level}"
            else:
                out_path = (
                    f"{DATASET_SAVE_DIR}/mc_{config_name_map[config_name]}_{level}"
                )

            if os.path.exists(out_path) and not OVERWRITE:
                print(f"Dataset {out_path} already exists, skipping...")
                continue
            if os.path.exists(out_path) and OVERWRITE:
                print(f"Overwriting existing dataset at {out_path}...")

            IMAGE_DIR = f"{DATA_ROOT_DIR}/{split_short}/{mode}/{split_short}_{config_name}"

            random.seed(42)  # same shuffling for all levels
            print(f"Creating MC: {level} dataset...")

            # Load meta.json only for adversarial levels
            meta_dict: Dict[int, dict] = {}
            if level != "clean":
                meta_json_path = os.path.join(
                    DATA_ROOT_DIR,
                    split_short,                     # "train" or "val"
                    mode,                            # "obj_attack"
                    f"{split_short}_{config_name}",  # e.g., "val_rc"
                    level,                           # "hard", "medium", "easy"
                    f"{mode}_{level}_meta.json",     # e.g., "obj_attack_easy_meta.json"
                )
                with open(meta_json_path, "r") as f:
                    raw_meta = json.load(f)
                # Convert keys to int and drop overlapping fields
                meta_dict = {}
                for qid_str, meta in raw_meta.items():
                    cleaned_meta = {
                        k: v for k, v in meta.items() if k not in META_DROP_KEYS
                    }
                    meta_dict[int(qid_str)] = cleaned_meta
                print(f"Loaded {len(meta_dict)} cleaned meta entries from {meta_json_path}")


            entries = []
            for i, item in enumerate(ds):
                if i % 1000 == 0:
                    print(f"  Processing {i}/{len(ds)}")
                image_id = item["image_id"]
                question_id = item["question_id"]

                mcq_item = mcq_data_dict[question_id]
                if mcq_item is None:
                    print(
                        f"Warning: question_id {question_id} not found in MCQ data, skipping."
                    )
                    continue

                # mcq_item example:
                # {
                #     "image_id": "003a8ae2ef43b901",
                #     "question_id": 34602,
                #     "answer": "Cassette deck",
                #     "choices": {
                #         "hard": "Remote control",
                #         "medium": "Spoon",
                #         "easy": "Watermelon"
                #     },
                #     ...
                # }

                choices = [
                    mcq_item["answer"],
                    mcq_item["choices"]["hard"],
                    mcq_item["choices"]["medium"],
                    mcq_item["choices"]["easy"],
                ]

                # Keep original behavior: no "clean" level appears here,
                # so we always use adversarial images for these levels.
                if level == "clean":
                    image = item["image"]
                    image = convert_img(image)  # ensure RGB JPEG format
                else:
                    image_path = glob.glob(
                        os.path.join(
                            IMAGE_DIR,
                            level,
                            "images",
                            f"{image_id}_{question_id}*",
                        )
                    )[0]
                    image = load_as_jpeg(image_path)

                if isinstance(image, dict) and "bytes" in image:
                    print(
                        f"Warning: question_id {question_id} image is in bytes dict format, converting to PIL Image."
                    )
                    exit()

                choices_shuffled = choices[:]
                random.shuffle(choices_shuffled)
                answer_letter = ["A", "B", "C", "D"][
                    choices_shuffled.index(mcq_item["answer"])
                ]
                question_text = QUESTION_FORMAT.format(
                    A=choices_shuffled[0],
                    B=choices_shuffled[1],
                    C=choices_shuffled[2],
                    D=choices_shuffled[3],
                )

                # Attach meta (does not change any existing fields/behavior)
                meta = None
                if level != "clean":
                    meta = meta_dict.get(question_id)

                entry = {
                    "image": image,
                    "question": question_text,
                    "answer": answer_letter,
                    "question_id": question_id,
                    "image_id": image_id,
                    "choices": {
                        "A": choices_shuffled[0],
                        "B": choices_shuffled[1],
                        "C": choices_shuffled[2],
                        "D": choices_shuffled[3],
                    },
                    "attack_word": ""
                    if level == "clean"
                    else (
                        mcq_item["answer"]
                        if level == "correct"
                        else mcq_item["choices"][level.split("-")[-1]]
                    ),
                    "meta": meta,  # new field
                }
                entries.append(entry)

            print(f"{level}: Created {len(entries)} entries.")
            dataset = Dataset.from_list(entries)
            dataset.save_to_disk(out_path)
            print(f"Saved {level} dataset to {out_path}")

        # ------------------------------------
        # --- Open-ended questions dataset ---
        # ------------------------------------
        # Load cosine-similarity info for label pruning
        qid2info_path = (
            f"{TEXTVQA_META_DIR}/textvqa_{split}_question_id2best_label.json"
        )
        with open(qid2info_path, "r") as f:
            qid2info = json.load(f)

        with open(f"{ASSETS_DIR}/abs_ancestors.pkl", "rb") as f:
            abs_ancestors = pickle.load(f)
        with open(f"{ASSETS_DIR}/all_ancestors.pkl", "rb") as f:
            all_ancestors = pickle.load(f)
        with open(f"{ASSETS_DIR}/parent2children.pkl", "rb") as f:
            parent2children = pickle.load(f)
        print("Loaded abs_ancestors, all_ancestors, parent2children")

        target_root = "Entity"

        def under_root(lbl: str) -> bool:
            if target_root is None:
                return True
            return target_root in abs_ancestors.get(lbl, {}).get(1, set())

        QUESTION_OE = (
            "What objects can be seen in the image? Answer only with object names."
        )

        for level in args.levels:
            if level == "clean":
                mode = "obj_clean"
            else:
                mode = "obj_attack"

            DATASET_SAVE_DIR = f"{DATA_DIR}/hf_dataset/{split_short}/{mode}"
            os.makedirs(DATASET_SAVE_DIR, exist_ok=True)

            if level == "clean":
                out_path = f"{DATASET_SAVE_DIR}/oe_{level}"
            else:
                out_path = (
                    f"{DATASET_SAVE_DIR}/oe_{config_name_map[config_name]}_{level}"
                )
            if os.path.exists(out_path) and not OVERWRITE:
                print(f"Dataset {out_path} already exists, skipping...")
                continue
            if os.path.exists(out_path) and OVERWRITE:
                print(f"Overwriting existing dataset at {out_path}...")

            IMAGE_DIR = f"{DATA_ROOT_DIR}/{split_short}/{mode}/{split_short}_{config_name}"

            # Load meta only for adversarial levels
            meta_dict: Dict[int, dict] = {}
            if level != "clean":
                meta_json_path = os.path.join(
                    DATA_ROOT_DIR,
                    split_short,
                    mode,
                    f"{split_short}_{config_name}",
                    level,
                    f"{mode}_{level}_meta.json",
                )
                with open(meta_json_path, "r") as f:
                    raw_meta = json.load(f)
                meta_dict = {}
                for qid_str, meta in raw_meta.items():
                    cleaned_meta = {
                        k: v for k, v in meta.items() if k not in META_DROP_KEYS
                    }
                    meta_dict[int(qid_str)] = cleaned_meta
                print(f"[OE] Loaded {len(meta_dict)} cleaned meta entries from {meta_json_path}")


            entries = []
            for i, item in enumerate(ds):
                if i % 1000 == 0:
                    print(f"  Processing {i}/{len(ds)}")
                image_id = item["image_id"]
                question_id = item["question_id"]

                mcq_item = mcq_data_dict[question_id]

                gt_labels = item["image_classes"]
                # Prune answers by prune_gt_to_pseudo_leaves
                gt_rooted = {g for g in gt_labels if under_root(g)} if gt_labels else set()
                gt_pruned = (
                    prune_gt_to_pseudo_leaves(gt_rooted, abs_ancestors)
                    if gt_rooted
                    else set()
                )
                # Filter by cosine similarity threshold
                info = qid2info[str(question_id)]
                info_pruned = {
                    k: v for k, v in info["all_scores"].items() if k in gt_pruned
                }
                info_pruned_sorted = dict(
                    sorted(info_pruned.items(), key=lambda x: x[1], reverse=True)
                )
                info_pruned_sorted = {
                    k: v for k, v in info_pruned_sorted.items() if v is not None
                }

                # Sanity check
                if len(info_pruned_sorted) == 0:
                    print(
                        f"Warning: question_id {question_id} has no valid pruned labels. Possible bug: include Aircraft class?"
                    )
                    # Bug case in original code: e.g., question_id 35274, all gt_label is "Aircraft"
                    # which is not under "Entity"
                    info_pruned_sorted = {
                        k: v
                        for k, v in info["all_scores"].items()
                        if k == "Aircraft"
                    }
                    if len(info_pruned_sorted) == 0:
                        exit()
                # None scores are removed in info_pruned_sorted
                for k, v in info_pruned_sorted.items():
                    if v is None:
                        print(
                            f"Warning: question_id {question_id} has None score for label {k}."
                        )
                        exit()

                # For open-ended, we do not need multiple-choice options here;
                # we just ensure that the answers come from the pruned label set.
                choices = [
                    mcq_item["answer"],
                    mcq_item["choices"]["hard"],
                    mcq_item["choices"]["medium"],
                    mcq_item["choices"]["easy"],
                ]

                if level == "clean":
                    image = item["image"]
                    image = convert_img(image)  # ensure RGB JPEG format
                else:
                    image_path = glob.glob(
                        os.path.join(
                            IMAGE_DIR,
                            level,
                            "images",
                            f"{image_id}_{question_id}*",
                        )
                    )[0]
                    image = load_as_jpeg(image_path)

                # Lookup meta for adversarial levels
                meta = None
                if level != "clean":
                    meta = meta_dict.get(question_id)

                answer2score_list = [
                    {"answer": k, "score": v} for k, v in info_pruned_sorted.items()
                ]
                entry = {
                    "image": image,
                    "question": QUESTION_OE,
                    "answers": list(info_pruned_sorted.keys()),
                    "answer2score": answer2score_list,
                    "question_id": question_id,
                    "image_id": image_id,
                    "attack_word": ""
                    if level == "clean"
                    else (
                        mcq_item["answer"]
                        if level == "correct"
                        else mcq_item["choices"][level]
                    ),
                    "meta": meta,  # new field
                }
                if i == 0:
                    print(f"Example entry: {entry}")
                entries.append(entry)

            print(f"{level}: Created {len(entries)} entries.")
            dataset = Dataset.from_list(entries)
            dataset.save_to_disk(out_path)
            print(f"Saved {level} open-ended dataset to {out_path}")
