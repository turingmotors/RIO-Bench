import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time

import torch
from PIL import Image, ImageDraw
try:
    from pytorch_lightning import seed_everything
except ModuleNotFoundError:
    def seed_everything(seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
from torch.utils.data import Dataset

from utils.completion_request import CompletionRequest
from utils.lingo_judge import LingoJudge
from utils.typo_attack_planner import AttackSkipError, TypoAttackPlanner, pil_to_base64
from utils.typo_attack_planner import format_instance_json
from utils.utils import is_correct_answer


class TypoDataset(Dataset):
    def __init__(self, question_path):
        # Question file
        with open(question_path, 'r') as f:
            self.questions = json.load(f)

    def __getitem__(self, index):
        line = self.questions[index]
        return line

    def __len__(self):
        return len(self.questions)


def create_data_loader(questions, batch_size=1, num_workers=0):
    dataset = TypoDataset(questions)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    return data_loader


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


def _export_hf_dataset_for_attack(args):
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
    cache_root = os.path.expanduser("~/.cache/huggingface/datasets")
    pattern = os.path.join(cache_root, "turing-motors___rio-bench", config_name, "*", "*", f"rio-bench-{{split}}*.arrow")
    matches = sorted(glob.glob(pattern))
    if matches:
        ds = Dataset.from_file(matches[-1])
        if "image" in ds.column_names:
            ds = ds.cast_column("image", HFImage())
    if ds is None:
        from datasets import load_from_disk
        local_roots = [
            os.path.expanduser("~/RIO-Bench/data/RIO-Bench/hf_dataset"),
            os.path.expanduser("~/RIO-Bench/data_patches/bugfix_1/RIO-Bench/hf_dataset"),
        ]
        # Try both dataset_name as-is and with __ replaced by / (e.g. txt_clean__original_with_suffix -> txt_clean/original_with_suffix)
        local_candidates = [dataset_name, os.path.join(split, config_name.replace("__", "/"))]
        for lr in local_roots:
            for cand in local_candidates:
                local_path = os.path.join(lr, cand)
                if os.path.exists(local_path):
                    ds = load_from_disk(local_path)
                    if "image" in ds.column_names:
                        ds = ds.cast_column("image", HFImage())
                    break
            if ds is not None:
                break
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
    question_id = ex.get("question_id", i)
    fname = image_id
    if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        fname = f"{{fname}}.jpg"
    path = os.path.join(images_dir, fname)
    if image_id not in seen:
        image.save(path)
        seen.add(image_id)
    record = dict(ex)
    record["image"] = fname
    record["text"] = ex.get("question")
    if "answer" not in record:
        answers = ex.get("answers")
        if isinstance(answers, list):
            vals = [str(a).strip() for a in answers if a is not None and str(a).strip()]
            record["answer"] = ", ".join(vals) if vals else ""
        elif isinstance(answers, str):
            record["answer"] = answers
        else:
            record["answer"] = ""
    record["question_id"] = question_id
    record["image_id"] = image_id
    records.append(record)

with open(meta_path, "w") as f:
    json.dump(records, f)
"""
    hf_python = args.hf_python if args.hf_python else sys.executable
    subprocess.check_call([hf_python, "-c", script])
    return export_dir, images_dir, meta_path


def _rect_to_pixels(rect, image_size):
    if rect is None or len(rect) != 4:
        return None
    W, H = image_size
    x, y, w, h = rect
    try:
        x = float(x)
        y = float(y)
        w = float(w)
        h = float(h)
    except Exception:
        return None
    if all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
        x0 = x * W
        y0 = y * H
        x1 = (x + w) * W
        y1 = (y + h) * H
    else:
        # Heuristic: if (w,h) look like bottom-right corners, use as x1,y1
        if w > x and h > y and w <= W and h <= H:
            x0, y0, x1, y1 = x, y, w, h
        else:
            x0, y0, x1, y1 = x, y, x + w, y + h
    x0 = max(0, min(W - 1, x0))
    y0 = max(0, min(H - 1, y0))
    x1 = max(0, min(W, x1))
    y1 = max(0, min(H, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _draw_ocr_bboxes_on_image(image: Image.Image, ocr_info, color=(255, 165, 0), width=2):
    if not ocr_info:
        return image
    out = image.copy()
    draw = ImageDraw.Draw(out)
    W, H = out.size
    for item in ocr_info:
        bb = item.get("bounding_box", {})
        if not {"top_left_x", "top_left_y", "width", "height"} <= bb.keys():
            continue
        try:
            x0 = float(bb["top_left_x"]) * W
            y0 = float(bb["top_left_y"]) * H
            x1 = float(bb["top_left_x"] + bb["width"]) * W
            y1 = float(bb["top_left_y"] + bb["height"]) * H
        except Exception:
            continue
        x0 = max(0, min(W - 1, int(round(x0))))
        y0 = max(0, min(H - 1, int(round(y0))))
        x1 = max(0, min(W, int(round(x1))))
        y1 = max(0, min(H, int(round(y1))))
        if x1 <= x0 or y1 <= y0:
            continue
        draw.rectangle((x0, y0, x1, y1), outline=color, width=width)
    return out


def _load_rio_txt_attack_word_map(split: str):
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "textvqa_misleading_word"))
    model_dir = os.path.join(base, "Llama-3.1-8B-Instruct")
    if split == "train":
        src = os.path.join(model_dir, "text_vqa_train_text_attack.json")
    else:
        src = os.path.join(model_dir, "text_vqa_validation_text_attack.json")
    if not os.path.exists(src):
        return {}, src
    with open(src, "r") as f:
        rows = json.load(f)
    qid2word = {}
    for row in rows:
        qid = row.get("question_id")
        attacks = row.get("attacks") or {}
        word = attacks.get("misleading")
        if qid is None or not isinstance(word, str) or not word.strip():
            continue
        qid2word[str(qid)] = word.strip()
    return qid2word, src


def _build_scenetap_dataset(args, mapping_path: str, fallback_map_path: str = ""):
    script = f"""
import glob
import json
import os
from datasets import Dataset, Image as HFImage, load_dataset

repo_id = {args.repo_id!r}
dataset_name = {args.dataset_name!r}
token = {args.hf_token!r} if {bool(args.hf_token)!r} else None
mapping_path = {mapping_path!r}
fallback_map_path = {fallback_map_path!r}
output_root = {args.output_dataset_root!r}
output_name = {args.output_dataset_name!r}

with open(mapping_path, "r") as f:
    mapping = json.load(f)
fallback_map = {{}}
if fallback_map_path and os.path.exists(fallback_map_path):
    with open(fallback_map_path, "r") as f:
        fallback_map = json.load(f)

split, config_name = dataset_name.split("/", 1)
ds = None
try:
    ds = load_dataset(repo_id, config_name, split=split, token=token)
except Exception:
    cache_root = os.path.expanduser("~/.cache/huggingface/datasets")
    pattern = os.path.join(cache_root, "turing-motors___rio-bench", config_name, "*", "*", f"rio-bench-{{split}}*.arrow")
    matches = sorted(glob.glob(pattern))
    if matches:
        ds = Dataset.from_file(matches[-1])
        if "image" in ds.column_names:
            ds = ds.cast_column("image", HFImage())
    if ds is None:
        raise

def _map(ex):
    qid = str(ex.get("question_id"))
    path = mapping.get(qid)
    fallback_flag = bool(fallback_map.get(qid, False))
    if path:
        return {{"image": path, "scenetap_fallback": fallback_flag}}
    return {{"image": ex.get("image"), "scenetap_fallback": fallback_flag}}

ds = ds.map(_map)
ds = ds.cast_column("image", HFImage())

if not output_name:
    output_name = f"{{config_name}}__scenetap"
out_dir = os.path.join(output_root, output_name, split)
os.makedirs(out_dir, exist_ok=True)
ds.save_to_disk(out_dir)
print(out_dir)
"""
    hf_python = args.hf_python if args.hf_python else sys.executable
    subprocess.check_call([hf_python, "-c", script])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name: gpt-4o")
    parser.add_argument("--dataset", type=str, default="typo_base_complex",
                        help="Dataset name: typo_base_complex, typo_base_color, vqav2_val2014")
    parser.add_argument("--attack", type=str, default="SceneTAP", help="Attack type: SceneTAP")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--log_dir", type=str, default="./log")
    parser.add_argument("--dataset_name", type=str, default="",
                        help="HF dataset name like 'val/obj_clean__mc_clean' (optional).")
    parser.add_argument("--repo_id", type=str, default="turing-motors/RIO-Bench",
                        help="HF dataset repo id.")
    parser.add_argument("--hf_token", type=str, default="",
                        help="HF token if required.")
    parser.add_argument("--hf_export_dir", type=str, default=".hf_export",
                        help="Fallback export dir when HF datasets is unavailable.")
    parser.add_argument("--hf_python", type=str, default="",
                        help="Python executable with HF datasets installed (fallback).")
    parser.add_argument("--save_hf_dataset", action="store_true",
                        help="Save SceneTAP images as a local HF dataset.")
    parser.add_argument("--output_dataset_root", type=str, default="./scenetap_hf",
                        help="Root dir to save local HF dataset.")
    parser.add_argument("--output_dataset_name", type=str, default="",
                        help="Output dataset name (default: <config>__scenetap).")
    parser.add_argument("--ocr_json_path", type=str, default="",
                        help="Path to TextVQA OCR JSON to avoid existing text.")
    parser.add_argument("--som_ref_dataset", type=str, default="",
                        help="Use SoM images from another dataset name (e.g., 'obj_clean__mc_clean').")
    parser.add_argument("--clean_base_dataset", type=str, default="",
                        help="Use images from this clean dataset as the attack base (e.g., 'val/obj_clean__mc_clean').")
    parser.add_argument("--save_sample_count", type=int, default=0,
                        help="If >0, copy up to N attacked images into a samples folder for quick inspection.")

    # chatgpt
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_tokens", type=int, default=4095)
    parser.add_argument("--top_p", type=float, default=0)

    # som
    parser.add_argument("--slider", type=float, default=2)
    parser.add_argument("--filter", type=float, default=None)

    # optional external adversarial text
    parser.add_argument("--adversarial-text", type=str, default=None,
                        help="Use a fixed adversarial text for all samples.")
    parser.add_argument("--adversarial-text-field", type=str, default=None,
                        help="Field name in question JSON to read per-sample adversarial text.")
    parser.add_argument("--caption-field", type=str, default=None,
                        help="Field name in question JSON to read per-sample caption text.")
    parser.add_argument("--use_attack_word", action="store_true",
                        help="Use per-sample attack_word from dataset if available.")
    parser.add_argument("--skip_model_eval", action="store_true",
                        help="Skip model answer generation/evaluation.")
    parser.add_argument("--skip_llm_plan", action="store_true",
                        help="Skip LLM planning; use a simple heuristic plan.")
    parser.add_argument("--question_id_start", type=int, default=None,
                        help="Process only samples with question_id >= this value.")
    parser.add_argument("--question_id_end", type=int, default=None,
                        help="Process only samples with question_id <= this value.")
    parser.add_argument("--question_id_mod", type=int, default=None,
                        help="Process only samples where question_id % mod == remainder.")
    parser.add_argument("--question_id_mod_remainder", type=int, default=0,
                        help="Remainder for question_id modulo filtering.")

    # seed
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # seed everything
    seed_everything(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.dataset_name:
        args.dataset = args.dataset_name.replace("/", "__")
    use_llm_plan = bool(os.environ.get("OPENAI_API_KEY")) and not args.skip_llm_plan
    plan_mode = "llm" if use_llm_plan else "bbox"

    # logger
    args.log_dir = os.path.join(args.log_dir, args.model, args.dataset, args.attack, f"plan_{plan_mode}")
    if args.attack == "SceneTAP":
        args.log_dir = os.path.join(args.log_dir, f"slider_{args.slider}", f"filter_{args.filter}")
    if args.question_id_start is not None or args.question_id_end is not None:
        qid_start = args.question_id_start if args.question_id_start is not None else "min"
        qid_end = args.question_id_end if args.question_id_end is not None else "max"
        args.log_dir = os.path.join(args.log_dir, f"qid_{qid_start}_{qid_end}")
    if args.question_id_mod is not None:
        args.log_dir = os.path.join(
            args.log_dir,
            f"qid_mod_{args.question_id_mod}_rem_{args.question_id_mod_remainder}",
        )

    args.log_dir = os.path.join(args.log_dir, f"seed_{args.seed}")
    os.makedirs(os.path.dirname(os.path.join(args.log_dir, "log.txt")), exist_ok=True)
    logger = logging.getLogger('test_logger')
    logger.setLevel(logging.DEBUG)
    log_path = os.path.join(args.log_dir, "log.txt")
    test_log = logging.FileHandler(f'{log_path}', 'a', encoding='utf-8')
    test_log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('')
    test_log.setFormatter(formatter)
    logger.addHandler(test_log)

    KZT = logging.StreamHandler()
    KZT.setLevel(logging.DEBUG)
    formatter = logging.Formatter('')
    KZT.setFormatter(formatter)
    logger.addHandler(KZT)

    logger.info("Config:")
    logger.info(json.dumps(args.__dict__, indent=2))
    logger.info("\n")

    # Path to save images
    image_save_dir = os.path.join(args.log_dir, "images")
    os.makedirs(image_save_dir, exist_ok=True)
    sample_save_dir = os.path.join(args.log_dir, "samples")
    if args.save_sample_count and args.save_sample_count > 0:
        os.makedirs(sample_save_dir, exist_ok=True)

    # Answer files
    answers_file = os.path.join(args.log_dir, "answer.jsonl")
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    existing_answer_list = []
    existing_answer_qids = set()
    if os.path.exists(answers_file):
        try:
            with open(answers_file, "r") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                existing_answer_list = loaded
                for item in existing_answer_list:
                    if isinstance(item, dict) and "question_id" in item:
                        existing_answer_qids.add(str(item["question_id"]))
        except Exception as e:
            logger.info(f"Warning: failed to load existing answer.jsonl, starting fresh. error={e}")

    # judgement model
    if args.dataset == "LingoQA":
        lingo_judge = LingoJudge()
    else:
        lingo_judge = None

    # Attack planner
    if args.attack == "SceneTAP":
        som_base_path = "./som_images"
        som_dataset = args.som_ref_dataset if args.som_ref_dataset else args.dataset
        if args.dataset_name and som_dataset in {"obj_clean__mc_clean", "obj_clean__oe_clean",
                                                "obj_attack__mc_easy", "obj_attack__mc_medium", "obj_attack__mc_hard",
                                                "obj_attack__oe_easy", "obj_attack__oe_medium", "obj_attack__oe_hard",
                                                "txt_clean__oe_clean", "txt_attack__oe_easy", "txt_attack__oe_hard"}:
            split_prefix = args.dataset_name.split("/", 1)[0]
            som_dataset = f"{split_prefix}__{som_dataset}"
        som_image_folder = os.path.join(som_base_path, som_dataset, f"slider_{args.slider}", f"seed_{args.seed}", f"filter_{args.filter}")
        typo_attack_planner = TypoAttackPlanner(som_image_folder, skip_llm_plan=not use_llm_plan)

    # Load the questions
    if args.dataset_name:
        _, export_images_dir, export_meta = _export_hf_dataset_for_attack(args)
        args.image_folder = export_images_dir
        args.question_file = export_meta

    questions = _load_questions(args.question_file)
    txt_attack_word_map = {}
    txt_attack_word_src = ""
    if args.attack == "SceneTAP" and args.use_attack_word and args.dataset_name and "txt_attack" in args.dataset_name:
        split = args.dataset_name.split("/", 1)[0]
        txt_attack_word_map, txt_attack_word_src = _load_rio_txt_attack_word_map(split)
        if txt_attack_word_map:
            logger.info(f"Loaded txt attack_word map: {len(txt_attack_word_map)} from {txt_attack_word_src}")
        else:
            logger.info(f"txt attack_word map is empty or missing: {txt_attack_word_src}")

    # Optional clean base images (for attack datasets)
    clean_image_id2path = {}
    if args.clean_base_dataset:
        class _Tmp:  # minimal args adapter for export
            pass
        tmp = _Tmp()
        tmp.repo_id = args.repo_id
        tmp.dataset_name = args.clean_base_dataset
        tmp.hf_token = args.hf_token
        tmp.hf_export_dir = args.hf_export_dir
        tmp.hf_python = args.hf_python
        _, clean_images_dir, clean_meta = _export_hf_dataset_for_attack(tmp)
        clean_records = _load_questions(clean_meta)
        for rec in clean_records:
            img = rec.get("image")
            img_id = rec.get("image_id")
            if img and img_id:
                clean_image_id2path[str(img_id)] = os.path.join(clean_images_dir, img)

    # OCR info (avoid existing text regions)
    image_id2ocr = {}
    if args.ocr_json_path:
        ocr_path = args.ocr_json_path
    elif args.dataset_name:
        split = args.dataset_name.split("/", 1)[0]
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "textvqa_meta"))
        if split == "train":
            ocr_path = os.path.join(base, "TextVQA_Rosetta_OCR_v0.2_train.json")
        else:
            ocr_path = os.path.join(base, "TextVQA_Rosetta_OCR_v0.2_val.json")
    else:
        ocr_path = ""
    if ocr_path and os.path.exists(ocr_path):
        with open(ocr_path, "r") as f:
            textocr_data = json.load(f)
        for item in textocr_data.get("data", []):
            image_id2ocr[item.get("image_id")] = item
        logger.info(f"Loaded OCR data: {ocr_path}")

    def _qid_to_int(value):
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return None

    ans_file_list = list(existing_answer_list)
    mapping_path = os.path.join(args.log_dir, "scenetap_mapping.json")
    fallback_map_path = os.path.join(args.log_dir, "scenetap_fallback_flags.json")
    existing_mapping = {}
    existing_fallback_flags = {}
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, "r") as f:
                loaded_mapping = json.load(f)
            if isinstance(loaded_mapping, dict):
                existing_mapping = loaded_mapping
        except Exception as e:
            logger.info(f"Warning: failed to load existing scenetap_mapping.json, starting fresh. error={e}")
    if os.path.exists(fallback_map_path):
        try:
            with open(fallback_map_path, "r") as f:
                loaded_fallback = json.load(f)
            if isinstance(loaded_fallback, dict):
                existing_fallback_flags = loaded_fallback
        except Exception as e:
            logger.info(f"Warning: failed to load existing scenetap_fallback_flags.json, starting fresh. error={e}")
    mapping = dict(existing_mapping)
    fallback_flags = dict(existing_fallback_flags)
    correct = 0
    total = 0
    sample_saved = 0

    def _append_answer_if_new(log_data):
        qid_key = str(log_data.get("question_id"))
        if qid_key in existing_answer_qids:
            return False
        ans_file_list.append(log_data)
        existing_answer_qids.add(qid_key)
        return True

    for i, data in enumerate(questions):
        placement_meta = {"fallback_used": False, "fallback_reason": None}
        question_id = data.get("question_id", i)
        qid_int = _qid_to_int(question_id)
        if args.question_id_start is not None or args.question_id_end is not None:
            if qid_int is None:
                logger.info({"question_id": question_id, "image": data.get("image"),
                             "skipped": True, "reason": "non_numeric_question_id"})
                continue
            if args.question_id_start is not None and qid_int < args.question_id_start:
                continue
            if args.question_id_end is not None and qid_int > args.question_id_end:
                continue
        if args.question_id_mod is not None:
            if qid_int is None:
                logger.info({"question_id": question_id, "image": data.get("image"),
                             "skipped": True, "reason": "non_numeric_question_id"})
                continue
            if qid_int % args.question_id_mod != args.question_id_mod_remainder:
                continue
        image_name = data["image"]
        image_id = data.get("image_id")
        if ("vqav2" in args.dataset or "LingoQA" in args.dataset) and args.attack != "no_attack":
            image_name_save = f"{image_name.split('.')[0]}_{question_id}.{image_name.split('.')[1]}"
            if args.attack != "SceneTAP":
                image_name = f"{image_name.split('.')[0]}_{question_id}.{image_name.split('.')[1]}"
        else:
            base, ext = os.path.splitext(image_name)
            image_name_save = f"{base}_{question_id}{ext}"
        image_path = os.path.join(args.image_folder, image_name)
        if args.clean_base_dataset and image_id in clean_image_id2path:
            image_path = clean_image_id2path[image_id]

        question = data.get("text", data.get("question", ""))
        correct_answer = data.get("answer", "")

        if args.attack == "SceneTAP":
            adversarial_text = None
            caption = None
            if args.adversarial_text:
                adversarial_text = args.adversarial_text
            elif args.use_attack_word:
                if data.get("attack_word"):
                    adversarial_text = data.get("attack_word")
                elif isinstance(data.get("attacks"), dict) and data.get("attacks", {}).get("misleading"):
                    adversarial_text = data.get("attacks", {}).get("misleading")
                elif isinstance(data.get("meta"), dict) and data.get("meta", {}).get("attack_word"):
                    adversarial_text = data.get("meta", {}).get("attack_word")
                elif txt_attack_word_map and str(question_id) in txt_attack_word_map:
                    adversarial_text = txt_attack_word_map[str(question_id)]
            elif args.adversarial_text_field and args.adversarial_text_field in data:
                adversarial_text = data[args.adversarial_text_field]
            if args.use_attack_word and (not isinstance(adversarial_text, str) or not adversarial_text.strip()):
                logger.info({"question_id": question_id, "image": data["image"],
                             "skipped": True, "reason": "missing_attack_word"})
                continue

            if args.caption_field and args.caption_field in data:
                caption = data[args.caption_field]

            ocr_info = image_id2ocr.get(image_id, {}).get("ocr_info", []) if image_id else []
            preferred_rect = None
            avoid_target_rect = None
            min_target_distance_px = 0.0
            with Image.open(image_path) as _img_probe:
                image_size = _img_probe.size
            if not use_llm_plan:
                meta = data.get("meta") or {}
                preferred_rect = _rect_to_pixels(meta.get("rect"), image_size)
            meta = data.get("meta") or {}
            img_w, img_h = image_size
            if args.dataset_name and "txt_attack" in args.dataset_name:
                avoid_target_rect = _rect_to_pixels(meta.get("answer_rect"), (img_w, img_h))
                min_target_distance_px = max(img_w, img_h) / 2.0
            som_seg_path = os.path.join(som_image_folder, image_name)
            som_mask_path = os.path.join(som_image_folder, image_name.replace(".jpg", ".npy"))
            if not (os.path.exists(som_seg_path) and os.path.exists(som_mask_path)):
                logger.info({"question_id": question_id, "image": data["image"],
                             "skipped": True, "reason": "missing_som_image_or_mask"})
                continue
            out_path = os.path.join(image_save_dir, f"{image_name_save}")
            if os.path.exists(out_path):
                logger.info({"question_id": question_id, "image": data["image"],
                             "skipped": True, "reason": "already_generated"})
                mapping[str(question_id)] = os.path.abspath(out_path)
                continue
            try:
                images, seg_image, plan_detail_origin, plan_detail, placement_meta = typo_attack_planner.attack(
                    image_path, question, correct_answer, adversarial_text=adversarial_text, caption=caption,
                    ocr_info=ocr_info, strict_ocr_avoid=True, preferred_rect=preferred_rect,
                    avoid_target_rect=avoid_target_rect, min_target_distance_px=min_target_distance_px)
            except AttackSkipError as e:
                logger.info({"question_id": question_id, "image": data["image"],
                             "skipped": True, "reason": str(e)})
                continue
            fallback_flags[str(question_id)] = bool(placement_meta.get("fallback_used", False))
            if placement_meta.get("fallback_used", False):
                logger.info({
                    "question_id": question_id,
                    "image": data["image"],
                    "fallback_used": True,
                    "fallback_reason": placement_meta.get("fallback_reason", "unknown"),
                })
            image = images[0]


            image.save(out_path)
            seg_out = _draw_ocr_bboxes_on_image(seg_image, ocr_info) if ocr_info else seg_image
            seg_out.save(os.path.join(image_save_dir, f"{image_name_save.replace('.jpg', '_seg.jpg')}"))
            mapping[str(question_id)] = os.path.abspath(os.path.join(image_save_dir, f"{image_name_save}"))
            if args.save_sample_count and sample_saved < args.save_sample_count:
                sample_out = os.path.join(sample_save_dir, f"{image_name_save}")
                if not os.path.exists(sample_out):
                    image.save(sample_out)
                sample_saved += 1
            # diffusion save path
            image_save_dir_diffusion = os.path.join(args.log_dir, "diffusion",
                                                    f"{image_name_save.replace('.jpg', '')}")
            os.makedirs(image_save_dir_diffusion, exist_ok=True)
            for k, img in enumerate(images):
                img.save(os.path.join(image_save_dir_diffusion, f"{k}.jpg"))
        else:
            image = Image.open(image_path).convert("RGB")
            images = [image]

        if args.skip_model_eval:
            log_data = {"question_id": question_id,
                        "image": data["image"],
                        "text": question,
                        "outputs": [],
                        "answer": correct_answer,
                        "plan_detail_origin": format_instance_json(
                            plan_detail_origin) if args.attack == "SceneTAP" else None,
                        "plan_detail": format_instance_json(plan_detail) if args.attack == "SceneTAP" else None,
                        "judge_list": [],
                        "fallback_used": bool(fallback_flags.get(str(question_id), False)) if args.attack == "SceneTAP" else None,
                        "fallback_reason": (placement_meta.get("fallback_reason")
                                            if args.attack == "SceneTAP" and fallback_flags.get(str(question_id), False)
                                            else None),
                        "is_correct": None
                        }
            _append_answer_if_new(log_data)
            logger.info(log_data)
            continue

        # Get the answer
        output_list = []
        judge_list = []
        for image in images:
            base64_image = pil_to_base64(image)
            completion_request = CompletionRequest(model=args.model, temperature=args.temperature,
                                                   max_tokens=args.max_tokens, top_p=args.top_p)
            completion_request.add_user_message_test(text=question, base64_image=[base64_image], image_first=True, detail="auto")
            completion = completion_request.get_completion_payload()
            answer = completion.choices[0].message.content

            answer = answer.lower()
            output_list.append(answer)
            judge_list.append(is_correct_answer(answer, correct_answer, question, args.dataset, lingo_judge=lingo_judge))

        # Save the answer
        if not(False in judge_list):
            correct += 1
            log_data = {"question_id": question_id,
                        "image": data["image"],
                        "text": question,
                        "outputs": output_list,
                        "answer": correct_answer,
                        "plan_detail_origin": format_instance_json(
                            plan_detail_origin) if args.attack == "SceneTAP" else None,
                        "plan_detail": format_instance_json(plan_detail) if args.attack == "SceneTAP" else None,
                        "judge_list": judge_list,
                        "fallback_used": bool(fallback_flags.get(str(question_id), False)) if args.attack == "SceneTAP" else None,
                        "fallback_reason": (placement_meta.get("fallback_reason")
                                            if args.attack == "SceneTAP" and fallback_flags.get(str(question_id), False)
                                            else None),
                        "is_correct": True
                        }
            _append_answer_if_new(log_data)
            logger.info(log_data)

        else:
            log_data = {"question_id": question_id,
                        "image": data["image"],
                        "text": question,
                        "outputs": output_list,
                        "answer": correct_answer,
                        "plan_detail_origin": format_instance_json(
                            plan_detail_origin) if args.attack == "SceneTAP" else None,
                        "plan_detail": format_instance_json(plan_detail) if args.attack == "SceneTAP" else None,
                        "judge_list": judge_list,
                        "fallback_used": bool(fallback_flags.get(str(question_id), False)) if args.attack == "SceneTAP" else None,
                        "fallback_reason": (placement_meta.get("fallback_reason")
                                            if args.attack == "SceneTAP" and fallback_flags.get(str(question_id), False)
                                            else None),
                        "is_correct": False
                        }
            _append_answer_if_new(log_data)
            logger.info(log_data)
        total += 1

    with open(answers_file, "w") as f:
        f.write(json.dumps(ans_file_list, indent=2))
    if not args.skip_model_eval:
        logger.info(f"Correct: {correct}/{total}")
        logger.info(f"Accuracy: {correct / total}")
        logger.info(f"ASR: {(total - correct) / total}")
    else:
        logger.info("Model evaluation skipped.")

    if args.save_hf_dataset and args.dataset_name:
        with open(mapping_path, "w") as f:
            json.dump(mapping, f)
        with open(fallback_map_path, "w") as f:
            json.dump(fallback_flags, f)
        _build_scenetap_dataset(args, mapping_path, fallback_map_path=fallback_map_path)
