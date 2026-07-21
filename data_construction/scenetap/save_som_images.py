import argparse
import json
import os
import random
import subprocess
import sys

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
try:
    from pytorch_lightning import seed_everything
except ModuleNotFoundError:
    def seed_everything(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
from utils.som import SoM


def _load_questions(path: str):
    if path.endswith(".jsonl"):
        items = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items
    with open(path, "r") as f:
        return json.load(f)


def _load_dataset_any(args):
    try:
        import datasets as hf_datasets  # HF datasets
        if not hasattr(hf_datasets, "load_dataset"):
            return None
        load_dataset = hf_datasets.load_dataset
        load_from_disk = hf_datasets.load_from_disk
    except Exception:
        return None

    if os.path.exists(args.dataset_name):
        return load_from_disk(args.dataset_name)
    if args.data_root:
        local_path = os.path.join(args.data_root, args.dataset_name)
        if os.path.exists(local_path):
            return load_from_disk(local_path)
    if "/" not in args.dataset_name:
        raise ValueError("Expected dataset_name like 'val/obj_clean__mc_clean' for Hub loading.")
    split, config_name = args.dataset_name.split("/", 1)
    token = args.hf_token if args.hf_token else None
    return load_dataset(args.repo_id, config_name, split=split, token=token)


def _get_image_from_entry(entry):
    image = entry.get("image")
    if image is None:
        image_path = entry.get("image_path")
        if image_path:
            return Image.open(image_path).convert("RGB")
        return None
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    return image


def _get_image_name(entry, fallback_idx: int):
    image_id = entry.get("image_id") or entry.get("image_name") or entry.get("image")
    if isinstance(image_id, dict):
        image_id = None
    if image_id is None:
        image_id = f"idx_{fallback_idx}"
    if isinstance(image_id, str) and os.path.sep in image_id:
        image_id = os.path.basename(image_id)
    name = str(image_id)
    if not name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        name = f"{name}.jpg"
    return name, str(image_id)


def _export_hf_dataset_via_python(args):
    export_root = os.path.abspath(args.hf_export_dir)
    export_name = args.dataset_name.replace("/", "__")
    export_dir = os.path.join(export_root, export_name)
    images_dir = os.path.join(export_dir, "images")
    meta_path = os.path.join(export_dir, "meta.json")
    if os.path.exists(meta_path) and os.path.isdir(images_dir):
        return export_dir, images_dir, meta_path

    os.makedirs(images_dir, exist_ok=True)
    script = f"""
import glob
import json
import os
from datasets import Dataset, Image as HFImage, load_dataset
from PIL import Image

repo_id = {args.repo_id!r}
dataset_name = {args.dataset_name!r}
token = {args.hf_token!r} if {bool(args.hf_token)!r} else None
export_dir = {export_dir!r}
images_dir = {images_dir!r}
meta_path = {meta_path!r}

split, config_name = dataset_name.split("/", 1)
ds = None
try:
    ds = load_dataset(repo_id, config_name, split=split, token=token)
except Exception:
    # Fallback: load cached arrow directly
    cache_root = os.path.expanduser("~/.cache/huggingface/datasets")
    pattern = os.path.join(cache_root, "turing-motors___rio-bench", config_name, "*", "*", f"rio-bench-{{split}}*.arrow")
    matches = sorted(glob.glob(pattern))
    if matches:
        ds = Dataset.from_file(matches[-1])
        if "image" in ds.column_names:
            ds = ds.cast_column("image", HFImage())
    if ds is None:
        raise

os.makedirs(images_dir, exist_ok=True)
seen = set()
records = []
for i, ex in enumerate(ds):
    image = ex.get("image")
    if image is None:
        continue
    image_id = ex.get("image_id") or ex.get("image_name") or f"idx_{{i}}"
    if isinstance(image_id, dict):
        image_id = None
    if image_id is None:
        image_id = f"idx_{{i}}"
    image_id = str(image_id)
    if image_id in seen:
        continue
    seen.add(image_id)
    fname = image_id
    if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        fname = f"{{fname}}.jpg"
    path = os.path.join(images_dir, fname)
    image.save(path)
    records.append({{"image": fname, "image_id": image_id}})

with open(meta_path, "w") as f:
    json.dump(records, f)
"""
    hf_python = args.hf_python if args.hf_python else sys.executable
    subprocess.check_call([hf_python, "-c", script])
    return export_dir, images_dir, meta_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="typo_base_complex",
                        help="Dataset name: typo_base_complex, typo_base_color, vqav2_val2014")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--dataset_name", type=str, default="",
                        help="HF dataset name like 'val/obj_clean__mc_clean' (optional).")
    parser.add_argument("--repo_id", type=str, default="turing-motors/RIO-Bench",
                        help="HF dataset repo id.")
    parser.add_argument("--hf_token", type=str, default="",
                        help="HF token if required.")
    parser.add_argument("--data_root", type=str, default="",
                        help="Local dataset root to resolve dataset_name (optional).")
    parser.add_argument("--hf_export_dir", type=str, default=".hf_export",
                        help="Fallback export dir when HF datasets is unavailable.")
    parser.add_argument("--hf_python", type=str, default="",
                        help="Python executable with HF datasets installed (fallback).")
    parser.add_argument("--log_dir", type=str, default="./som_images")
    parser.add_argument("--filter", type=float, default=10.0)

    # som
    parser.add_argument("--slider", type=float, default=3)

    # seed
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.dataset_name and args.dataset == "typo_base_complex":
        args.dataset = args.dataset_name.replace("/", "__")

    # seed everything
    seed_everything(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # logger
    args.log_dir = os.path.join(args.log_dir, args.dataset, f"slider_{args.slider}")
    args.log_dir = os.path.join(args.log_dir, f"seed_{args.seed}", f"filter_{args.filter}")
    os.makedirs(args.log_dir, exist_ok=True)

    if args.dataset_name:
        dataset = _load_dataset_any(args)
        questions = None
        if dataset is None:
            export_dir, images_dir, meta_path = _export_hf_dataset_via_python(args)
            args.image_folder = images_dir
            questions = _load_questions(meta_path)
    else:
        questions = _load_questions(args.question_file)
        dataset = None

    # SoM
    semsam_cfg = "SoM/configs/semantic_sam_only_sa-1b_swinL.yaml"
    seem_cfg = "SoM/configs/seem_focall_unicl_lang_v1.yaml"
    semsam_ckpt = "SoM/swinl_only_sam_many2many.pth"
    sam_ckpt = "SoM/sam_vit_h_4b8939.pth"
    seem_ckpt = "SoM/seem_focall_v1.pt"

    som = SoM(semsam_cfg, seem_cfg, semsam_ckpt, sam_ckpt, seem_ckpt)
    image_set = set()
    if dataset is not None:
        for idx, entry in enumerate(tqdm(dataset, desc="SoM", unit="img")):
            image = _get_image_from_entry(entry)
            if image is None:
                continue
            image_name, image_id = _get_image_name(entry, idx)
            if image_id in image_set:
                continue
            image_set.add(image_id)
            out_img_path = os.path.join(args.log_dir, image_name)
            out_mask_path = os.path.join(args.log_dir, os.path.splitext(image_name)[0] + '.npy')
            if os.path.exists(out_img_path) and os.path.exists(out_mask_path):
                continue

            # get segmentation image and map
            seg_image, mask = som.inference(image=image, slider=args.slider, mode="Automatic", alpha=0.2,
                                                 label_mode="Number",
                                                 anno_mode=['Mask', 'Mark'], filter=args.filter)
            seg_image = Image.fromarray(seg_image)
            seg_image = seg_image.resize(image.size)
            seg_image.save(out_img_path)
            # save the mask to .npy file, mask it now a list of numpy arrays
            np.save(out_mask_path, mask)
    else:
        for entry in tqdm(questions, desc="SoM", unit="img"):
            image_name = entry["image"]
            if image_name in image_set:
                continue
            image_set.add(image_name)
            image_path = os.path.join(args.image_folder, image_name)
            # Load the image
            image = Image.open(image_path).convert("RGB")
            out_img_path = os.path.join(args.log_dir, image_name)
            out_mask_path = os.path.join(args.log_dir, image_name.split('.')[0] + '.npy')
            if os.path.exists(out_img_path) and os.path.exists(out_mask_path):
                continue

            # get segmentation image and map
            seg_image, mask = som.inference(image=image, slider=args.slider, mode="Automatic", alpha=0.2,
                                                 label_mode="Number",
                                                 anno_mode=['Mask', 'Mark'], filter=args.filter)
            seg_image = Image.fromarray(seg_image)
            seg_image = seg_image.resize(image.size)
            seg_image.save(out_img_path)
            # save the mask to .npy file, mask it now a list of numpy arrays
            np.save(out_mask_path, mask)
