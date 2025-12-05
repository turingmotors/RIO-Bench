# Create Obj-Clean and Obj-Attack datasets by combining multiple-choice questions and adversarial examples
import os
import glob
import json
from typing import Dict, List, Set, Tuple, Optional
from datasets import load_dataset, Dataset, DatasetDict
import random
import pickle   
from PIL import Image
from datasets import Image as HFImage
import numpy as np

import io
from PIL import Image as PILImage

from obj_create_multi_choice_data import prune_gt_to_pseudo_leaves


# -------------------------------------------
# split = "validation"
split = "train"
split_short = "val" if split == "validation" else "train"
# -------------------------------------------


# Load original TextVQA validation set
ds = load_dataset("facebook/textvqa", split=split)
print(f"Loaded TextVQA validation set with {len(ds)} entries.")

# Load multiple-choice questions
mcq_json_path = f"textvqa_{split_short}_mcq_samples.json"
with open(mcq_json_path, "r") as f:
    mcq_data = json.load(f)
print(f"Loaded {len(mcq_data)} multiple-choice questions from {mcq_json_path}")
mcq_data_dict = {item["question_id"]: item for item in mcq_data}

# format
# {'answers': ['lape',
#              'cargo',
#              'ec-agg',
#              'lape',
#              'lape',
#              'lape',
#              'lape',
#              'lape',
#              'lape',
#              'airplane'],
#  'flickr_300k_url': 'https://c8.staticflickr.com/3/2579/5811451782_f6c4633327_z.jpg',
#  'flickr_original_url': 'https://c8.staticflickr.com/3/2579/5811451782_31f8649055_o.jpg',
#  'image': <PIL.JpegImagePlugin.JpegImageFile image mode=L size=1024x667 at 0x1554D87F91C0>,
#  'image_classes': ['Vehicle', 'Helicopter', 'Airplane', 'Bomb', 'Aircraft'],
#  'image_height': 667,
#  'image_id': '005635e119b9f32f',
#  'image_width': 1024,
#  'question': 'what type of plane is this?',
#  'question_id': 1,
#  'question_tokens': ['what', 'type', 'of', 'plane', 'is', 'this'],
#  'set_name': 'train'}

def convert_img(img: Image.Image, quality=95) -> Image.Image:
    """Convert image to RGB JPEG format in memory."""
    # RGB, JPEG, Quality
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf)

def load_as_jpeg(path: str, quality: int = 95):
    """PNGでもJPEGに変換してRGBで読み込む"""
    with Image.open(path) as img:
        return convert_img(img, quality=quality)


mode = "obj_attack"

DATASET_SAVE_DIR = f"../data/RIO-Bench/hf_dataset/{split_short}/{mode}"
os.makedirs(DATASET_SAVE_DIR, exist_ok=True)
config_name_list = ["rc"] #, "grid_white_28px"]
config_name_map = {
    "rc": "random",
}
for config_name in config_name_list:
    print(f"Processing config: {config_name}")
    IMAGE_DIR = f"../data/RIO-Bench/{split_short}/{mode}/{config_name}"

    # ------------------------------------------
    # --- Multiple-choice questions dataset ---
    # ------------------------------------------
    QUESTION_FORMAT = "Which object is present in the image? (A) {A} (B) {B} (C) {C} (D) {D}. Answer with only the option letter (A, B, C, or D)."
    random.seed(42)  # for reproducibility
    for level in ["correct", "hard", "medium", "easy"]:
        print(f"Creating MCP: {level} dataset...")
        entries = []
        for i, item in enumerate(ds):
            if i % 1000 == 0:
                print(f"  Processing {i}/{len(ds)}")
            image_id = item["image_id"]
            question_id = item["question_id"]
            
            mcq_item = mcq_data_dict[question_id]
            if mcq_item is None:
                print(f"Warning: question_id {question_id} not found in MCQ data, skipping.")
                continue

            # mcq_item: {
            #     "image_id": "003a8ae2ef43b901",
            #     "question_id": 34602,
            #     "answer": "Cassette deck",
            #     "choices": {
            #     "hard": "Remote control",
            #     "medium": "Spoon",
            #     "easy": "Watermelon"
            #     },
            #     "image_classes": [
            #     "Cassette deck",
            #     "Printer",
            #     "Medical equipment",
            #     "Computer mouse",
            #     "Scale",
            #     "Telephone",
            #     "Camera",
            #     "Ipod",
            #     "Remote control"
            #     ]
            # },

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
                image_path = glob.glob(os.path.join(IMAGE_DIR, level, "images", f"{image_id}_{question_id}*"))[0]
                image = load_as_jpeg(image_path)
            

            if isinstance(image, dict) and "bytes" in image:
                print(f"Warning: question_id {question_id} image is in bytes dict format, converting to PIL Image.")
                exit()

            choices_shuffled = choices[:]
            random.shuffle(choices_shuffled)
            answer_letter = ["A", "B", "C", "D"][choices_shuffled.index(mcq_item["answer"])]
            question_text = QUESTION_FORMAT.format(A=choices_shuffled[0], B=choices_shuffled[1], C=choices_shuffled[2], D=choices_shuffled[3])

            entry = {
                "image": image,
                "question": question_text,
                "answer": answer_letter,
                "question_id": question_id,
                "image_id": image_id,
                "choices": {"A": choices_shuffled[0], "B": choices_shuffled[1], "C": choices_shuffled[2], "D": choices_shuffled[3]},
                "attack_word": "" if level == "clean" else mcq_item["choices"][level.split('-')[-1]],
            }
            entries.append(entry)

        print(f"{level}: Created {len(entries)} entries.")
        dataset = Dataset.from_list(entries)
        if level == "clean":
            out_path = f"{DATASET_SAVE_DIR}/mcq_{level}"
        else:
            out_path = f"{DATASET_SAVE_DIR}/mcq_{config_name_map[config_name]}_{level}"
        dataset.save_to_disk(out_path)
        print(f"Saved {level} dataset to {out_path}")

    # ------------------------------------
    # --- Open-ended questions dataset ---
    # ------------------------------------
    # calc X% cosine similarity threshold 
    qid2info_path = f"../data_construction/textvqa_{split_short}_question_id2best_label.json"
    with open(qid2info_path, "r") as f:
        qid2info = json.load(f)

    ASSETS_DIR = "assets/open_images"
    with open(f"{ASSETS_DIR}/abs_ancestors.pkl", "rb") as f:
        abs_ancestors = pickle.load(f)
    with open(f"{ASSETS_DIR}/all_ancestors.pkl", "rb") as f:
        all_ancestors = pickle.load(f)
    with open(f"{ASSETS_DIR}/parent2children.pkl", "rb") as f:
        parent2children = pickle.load(f)
    print("loaded abs_ancestors, all_ancestors, parent2children")

    target_root = "Entity"
    def under_root(lbl: str) -> bool:
        if target_root is None:
            return True
        return target_root in abs_ancestors.get(lbl, {}).get(1, set())

    PROMPT_TAG = "v3"
    QUESTION_OE = "What objects can be seen in the image? Answer only with object names."
    for level in ["clean", "hard", "medium", "easy"]:
        entries = []
        for i, item in enumerate(ds):
            if i % 1000 == 0:
                print(f"  Processing {i}/{len(ds)}")
            image_id = item["image_id"]
            question_id = item["question_id"]

            mcq_item = mcq_data_dict[question_id]

            gt_labels = item["image_classes"]
            # prune answers by prune_gt_to_pseudo_leaves
            gt_rooted = {g for g in gt_labels if under_root(g)} if gt_labels else set()
            gt_pruned = prune_gt_to_pseudo_leaves(gt_rooted, abs_ancestors) if gt_rooted else set()
            # filter by cosine similarity threshold
            info = qid2info[str(question_id)]
            info_pruned = {k: v for k, v in info["all_scores"].items() if k in gt_pruned}
            info_pruned_sorted = dict(sorted(info_pruned.items(), key=lambda x: x[1], reverse=True))
            info_pruned_sorted = {k: v for k, v in info_pruned_sorted.items() if v is not None}

            # sanity check
            if len(info_pruned_sorted) == 0:
                print(f"Warning: question_id {question_id} has no valid pruned labels. Bug: include Aircraft class?")
                # bug: for question_id 35274, all gt_label is "Aircraft" which is not under "Entity"
                info_pruned_sorted = {k: v for k, v in info["all_scores"].items() if k == "Aircraft"}
                if len(info_pruned_sorted) == 0:
                    exit()
            # None scores are removed in info_pruned_sorted
            for k, v in info_pruned_sorted.items():
                if v is None:
                    print(f"Warning: question_id {question_id} has None score for label {k}.")
                    exit()

            # For open-ended, we don't need choices, just ensure the answer is in gt_final
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
                image_path = glob.glob(os.path.join(IMAGE_DIR, level, "images", f"{image_id}_{question_id}*"))[0]
                image = load_as_jpeg(image_path)

            entry = {
                "image": image,
                "question": QUESTION_OE,
                "answers": list(info_pruned_sorted.keys()),
                "answer2score": info_pruned_sorted,
                "question_id": question_id,
                "image_id": image_id,
                "attack_word": "" if level == "clean" else mcq_item["choices"][level],
            }
            entries.append(entry)

            if i == 0:
                print(f"Example entry: {entry}")

        print(f"{level}: Created {len(entries)} entries.")
        dataset = Dataset.from_list(entries)
        if level == "clean":
            out_path = f"{DATASET_SAVE_DIR}/open_ended_{level}"
        else:
            out_path = f"{DATASET_SAVE_DIR}/open_ended_{config_name_map[config_name]}_{level}"
        if PROMPT_TAG != "v1":
            out_path += f"_{PROMPT_TAG}"
        dataset.save_to_disk(out_path)
        print(f"Saved {level} open-ended dataset to {out_path}")
