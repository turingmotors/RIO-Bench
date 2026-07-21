from typo_attack_utils import apply_pattern_for_image
from typing import List, Dict, Tuple, Set, Union, Iterable
from PIL import Image
import PIL
import random
import os
from tqdm import tqdm

DEFAULT_GRID_ORDER = ("C","TL","TR","BL","BR","TC","BC","CL","CR")

if __name__ == "__main__":
    import argparse
    from datasets import load_dataset
    import json
    import random

    parser = argparse.ArgumentParser(description="Generate txt typo-attack images.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["validation", "train"],
        default=["validation", "train"],
        help="TextVQA splits to process.",
    )
    parser.add_argument(
        "--choice-levels",
        nargs="+",
        choices=["misleading", "correct"],
        default=["misleading", "correct"],
        help="Attack word level(s) to generate.",
    )
    args = parser.parse_args()

    # -----------------------------
    # Common debug settings
    # -----------------------------
    IS_ONLY_DEBUG = False
    N_DEBUG_IMAGES = 10  # per choice level

    # -----------------------------
    # Common config / base dirs
    # -----------------------------
    CONFIG_DIR = "typo_attack_config"
    BASE_DIR = "../data/"
    MISLEADING_WORD_DIR = "assets/textvqa_misleading_word/Llama-3.1-8B-Instruct"

    # -----------------------------
    # Process both validation and train
    # -----------------------------
    split_configs = [
        (
            "validation",
            "val",
            [
                os.path.join(CONFIG_DIR, "txt_attack", "val_config_far.json"),
                os.path.join(CONFIG_DIR, "txt_attack", "val_config_mid.json"),
            ],
        ),
        (
            "train",
            "train",
            [
                os.path.join(CONFIG_DIR, "txt_attack", "train_config_far.json"),
                os.path.join(CONFIG_DIR, "txt_attack", "train_config_mid.json"),
            ],
        ),
    ]
    for split, split_short, cfg_paths in split_configs:
        if split not in set(args.splits):
            continue
        print(f"Generating text-attack images for TextVQA {split} set")

        # ---- load TextVQA split ----
        ds = load_dataset("facebook/textvqa", split=split)

        # ---- load OCR data of TextVQA ----
        textvqa_ocr_json = f"assets/textvqa_meta/TextVQA_Rosetta_OCR_v0.2_{split_short}.json"
        with open(textvqa_ocr_json, "r") as f:
            textvqa_ocr_data = json.load(f)
        image_id2ocr = {}
        for idx, item in enumerate(textvqa_ocr_data["data"]):
            image_id2ocr[item["image_id"]] = item

        # ---- load misleading choices data ----
        misleading_txt_json = f"{MISLEADING_WORD_DIR}/text_vqa_{split}_text_attack.json"
        with open(misleading_txt_json, "r") as f:
            misleading_txt_data = json.load(f)
        print(f"Loaded {misleading_txt_json}")
        misleading_txt_data = {item["question_id"]: item for item in misleading_txt_data}

        # validation-only flag for DEBUG_IMAGE_ID behavior
        if split == "validation":
            is_debug_image_id_shown = False

        # ---- config files (per split) ----
        for cfg_path in cfg_paths:
            assert os.path.exists(cfg_path), f"Config file not found: {cfg_path}"
            print(f"Using config files: {cfg_paths}")
            with open(cfg_path, "r") as f:
                config = json.load(f)
            print(f"Loaded config from {cfg_path}")
            pat = config["patterns"][0]

            # Typographic attack for TextVQA
            OUT_DIR = os.path.join(BASE_DIR, split_short, "txt_attack")
            os.makedirs(OUT_DIR, exist_ok=True)

            random.seed(42)  # for reproducibility
            choice_levels = list(args.choice_levels)
            for choice_level in choice_levels:
                print(f"Generating text-attack images for choice level: {choice_level}")

                # Keep original path/layout for "misleading" to avoid changing existing behavior.
                if choice_level == "misleading":
                    THIS_OUT_DIR = os.path.join(OUT_DIR, pat["pattern_id"])
                elif choice_level == "correct":
                    THIS_OUT_DIR = os.path.join(OUT_DIR, pat["pattern_id"], "correct")
                else:
                    raise ValueError(f"Unknown choice level: {choice_level}")

                os.makedirs(THIS_OUT_DIR, exist_ok=True)
                THIS_IMAGE_DIR = os.path.join(THIS_OUT_DIR, "images")
                os.makedirs(THIS_IMAGE_DIR, exist_ok=True)

                meta_info = {}
                for i, item in tqdm(enumerate(ds), total=len(ds)):
                    if i % 100 == 0:
                        print(f"Processing {i}/{len(ds)}")

                    image = item["image"]
                    image_id = item["image_id"]
                    question = item["question"]
                    question_id = item["question_id"]
                    ocr_info = image_id2ocr[image_id].get("ocr_info", [])
                    answers = item["answers"]
                    # most frequent answer
                    answer = max(set(answers), key=answers.count)

                    mt_item = misleading_txt_data[question_id]
                    if choice_level == "misleading":
                        attack_word = mt_item["attacks"]["misleading"]
                    elif choice_level == "correct":
                        attack_word = answer
                    else:
                        raise ValueError(f"Unknown choice level: {choice_level}")

                    # text-attack (non-debug)
                    if not IS_ONLY_DEBUG:
                        res = apply_pattern_for_image(
                            image=image,
                            ocr_info=ocr_info,
                            text_to_place=attack_word,
                            pattern=pat,
                            config=config,
                            image_id=image_id,
                            question_id=question_id,
                            question_text=question,
                            subset_name=split,
                            answer_text=answer,
                        )

                        out_img = res["image"]
                        meta = res["metadata"]

                        fname = (
                            f"{image_id}_{question_id}_{split}"
                            f"__{meta['pattern_id']}__seed{meta['seed']}.jpg"
                        )
                        out_path = os.path.join(THIS_IMAGE_DIR, fname)
                        out_img.save(out_path)
                        # save meta info
                        meta["attack_word"] = attack_word
                        meta["answer"] = answer
                        meta_info[question_id] = meta

                        if i % 100 == 0:
                            print(meta_info[question_id])

                    if i < N_DEBUG_IMAGES:
                        res = apply_pattern_for_image(
                            image=image,
                            ocr_info=ocr_info,
                            text_to_place=attack_word,
                            pattern=pat,
                            config=config,
                            image_id=image_id,
                            question_id=question_id,
                            question_text=question,
                            subset_name=split,
                            answer_text=answer,
                            is_debug=True,
                        )
                        debug_img = res["image"]
                        debug_meta = res["metadata"]
                        debug_fname = f"{image_id}_{question_id}_{split}__debug.jpg"
                        DEBUG_IMAGE_DIR = os.path.join(THIS_OUT_DIR, "debug_images")
                        os.makedirs(DEBUG_IMAGE_DIR, exist_ok=True)
                        debug_out_path = os.path.join(DEBUG_IMAGE_DIR, debug_fname)
                        debug_img.save(debug_out_path)
                        print(f"Saved debug image to {debug_out_path}")

                # Keep original metadata filename for "misleading"
                if not IS_ONLY_DEBUG:
                    if choice_level == "misleading":
                        meta_json_name = f"text_attack_{pat['pattern_id']}_meta.json"
                    else:
                        meta_json_name = f"text_attack_{pat['pattern_id']}_{choice_level}_meta.json"

                    meta_json_path = os.path.join(THIS_OUT_DIR, meta_json_name)
                    with open(meta_json_path, "w") as f:
                        json.dump(meta_info, f, indent=2)
                    print(
                        f"Saved metadata for {pat['pattern_id']} ({choice_level}) "
                        f"text-attack to {meta_json_path}"
                    )
