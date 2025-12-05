# Create Obj-Clean and Obj-Attack datasets by combining multiple-choice questions and adversarial examples
# 1. Append a suffix to each question to encourage concise answers
# 2. Convert all images to RGB JPEG format in memory
# 3. Save the processed datasets to disk

import os
import json
import glob
from typing import Dict, List, Set, Tuple, Optional
from datasets import load_dataset, Dataset, DatasetDict
import random
import pickle   
from PIL import Image
from datasets import Image as HFImage

import io
from PIL import Image as PILImage

import copy

from obj_create_multi_choice_data import prune_gt_to_pseudo_leaves


SUFFIX = " Answer the question using a single word or phrase."

#---------------------------
# split = "validation"
# split_short = "val"
# method_list = ["txt_bucket_val_far", "txt_bucket_val_mid"]
# method_map = {
#     "txt_bucket_val_far": "far",
#     "txt_bucket_val_mid": "mid",
# }


split = "train"
split_short = "train"
method_list = ["txt_bucket_train_far", "txt_bucket_train_mid"]
method_map = {
    "txt_bucket_train_far": "far",
    "txt_bucket_train_mid": "mid",
}
#---------------------------


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


# Load original TextVQA validation set
ds = load_dataset("facebook/textvqa", split=split)
print(f"Loaded TextVQA {split} set with {len(ds)} entries.")

# save to local
entries = []
for i, item in enumerate(ds):
    if i % 1000 == 0:
        print(f"  Processing {i}/{len(ds)}")
    # Add suffix to question
    question = item["question"]
    item["question"] = question + SUFFIX

    # Convert image to RGB JPEG format
    item["image"] = convert_img(item["image"])
    entries.append(item)
dataset = Dataset.from_list(entries)
out_path = f"../data/RIO-Bench/hf_dataset/{split_short}/txt_clean/original_with_suffix"
dataset.save_to_disk(out_path)
print(f"Saved original TextVQA validation set to {out_path}")

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


DATASET_SAVE_DIR = f"../data/RIO-Bench/hf_dataset/{split_short}/txt_attack"
IMAGE_DIR = f"../data/RIO-Bench/{split_short}/txt_attack/"

for method in method_list:
    # ------------------------------------------
    # --- Multiple-choice questions dataset ---
    # ------------------------------------------
    print(f"Creating MCP: {method} dataset...")
    entries = []
    for i, item in enumerate(ds):
        if i % 1000 == 0:
            print(f"  Processing {i}/{len(ds)}")
        image_id = item["image_id"]
        question_id = item["question_id"]

        image_path = glob.glob(os.path.join(IMAGE_DIR, method, "images", f"{image_id}_{question_id}*"))[0]
        image = load_as_jpeg(image_path)
        if isinstance(image, dict) and "bytes" in image:
            print(f"Warning: question_id {question_id} image is in bytes dict format, converting to PIL Image.")
            exit()

        entry = copy.deepcopy(item)
        entry["image"] = image

        question = entry["question"]
        entry["question"] = question + SUFFIX

        entries.append(entry)

    print(f"{method}: Created {len(entries)} entries.")
    dataset = Dataset.from_list(entries)
    out_path = f"{DATASET_SAVE_DIR}/open_ended_{method_map[method]}"
    dataset.save_to_disk(out_path)
    print(f"Saved {method} dataset to {out_path}")