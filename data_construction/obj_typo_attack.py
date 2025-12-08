from typo_attack_utils import apply_pattern_for_image
from typing import Dict
from PIL import Image
import PIL
import random
import os
import json
from datasets import load_dataset

# Default grid order used by some placement patterns (kept for reference)
DEFAULT_GRID_ORDER = ("C", "TL", "TR", "BL", "BR", "TC", "BC", "CL", "CR")


def generate_obj_attack_for_split(
    split: str,
    hf_split: str,
    config_path: str,
    ocr_json_path: str,
    mc_json_path: str,
    choice_levels,
    base_dir: str = "../data/",
    is_only_debug: bool = False,
    n_debug_images: int = 0,
    debug_image_id: str | None = None,
) -> None:
    """
    Generate object-attack images for a given split (e.g., 'val' or 'train').

    Args:
        split:        "val" or "train" (used for output dirs and filenames).
        hf_split:     HuggingFace datasets split name ("validation" or "train").
        config_path:  Path to typo-attack config JSON.
        ocr_json_path:Path to TextVQA OCR JSON.
        mc_json_path: Path to MCQ metadata JSON.
        choice_levels:Iterable of choice levels, e.g. ["correct"] or ["easy","medium","hard"].
        base_dir:     Base directory to store generated images and metadata.
        is_only_debug:If True, only debug images are generated and main outputs are skipped.
        n_debug_images:
            For "val": maximum number of debug images by index (i < n_debug_images).
            For "train": this argument is ignored; the first 5 samples are used for debug,
                         as in the original code.
        debug_image_id:
            For "val": if not None, a specific image_id that forces a debug image and then
                       stops further debug generation when seen (original behavior).
            For "train": ignored.
    """
    print(f"Generating obj-attack images for TextVQA {split} set")

    # Load typo-attack config
    with open(config_path, "r") as f:
        config = json.load(f)
    print(f"Loaded config from {config_path}")
    pat = config["patterns"][0]

    # Load TextVQA split
    ds = load_dataset("facebook/textvqa", split=hf_split)

    # Load OCR data
    with open(ocr_json_path, "r") as f:
        textocr_data = json.load(f)
    print(f"Loaded {ocr_json_path}")

    # Load multiple-choice metadata
    with open(mc_json_path, "r") as f:
        mc_data = json.load(f)
    mc_data_dict = {item["question_id"]: item for item in mc_data}

    # Map image_id -> OCR annotation
    image_id2ocr: Dict[str, Dict] = {}
    for item in textocr_data["data"]:
        image_id2ocr[item["image_id"]] = item

    out_dir = os.path.join(base_dir, split, "obj_attack")
    os.makedirs(out_dir, exist_ok=True)

    # Flag used only for the "val" split to stop debug images after a specific image_id
    is_debug_image_id_shown = False

    for choice_level in choice_levels:
        random.seed(42)  # same attack config across levels

        print(f"Generating obj-attack images for choice level: {choice_level}")
        this_out_dir = os.path.join(out_dir, pat["pattern_id"], choice_level)
        os.makedirs(this_out_dir, exist_ok=True)

        this_image_dir = os.path.join(this_out_dir, "images")
        os.makedirs(this_image_dir, exist_ok=True)

        meta_info: Dict[str, Dict] = {}

        for i, item in enumerate(ds):
            if i % 100 == 0:
                print(f"Processing {i}/{len(ds)}")

            image = item["image"]
            image_id = item["image_id"]
            question = item["question"]
            question_id = item["question_id"]
            ocr_info = image_id2ocr.get(image_id, {}).get("ocr_info", [])

            mc_item = mc_data_dict[question_id]
            answer = mc_item["answer"]
            choices = mc_item["choices"]
            c_hard = choices["hard"]
            c_medium = choices["medium"]
            c_easy = choices["easy"]

            # Determine which word to overlay depending on choice level
            if choice_level == "hard":
                attack_word = c_hard
            elif choice_level == "medium":
                attack_word = c_medium
            elif choice_level == "easy":
                attack_word = c_easy
            elif choice_level == "correct":
                attack_word = answer
            else:
                raise ValueError(f"Unknown choice level: {choice_level}")

            # Main obj-attack image generation (non-debug)
            if not is_only_debug:
                res = apply_pattern_for_image(
                    image=image,
                    ocr_info=ocr_info,
                    text_to_place=attack_word,
                    pattern=pat,
                    config=config,
                    image_id=image_id,
                    question_id=question_id,
                    subset_name=split,
                    answer_text=(
                        answer
                        if pat.get("placement", {}).get("mode", "")
                        == "grid3x3_bucketed_simple"
                        else None
                    ),
                )

                out_img = res["image"]
                meta = res["metadata"]

                fname = (
                    f"{image_id}_{question_id}_{split}__"
                    f"{meta['pattern_id']}__seed{meta['seed']}.jpg"
                )
                out_path = os.path.join(this_image_dir, fname)
                out_img.save(out_path)

                # Save metadata for this attacked sample
                meta["attack_word"] = attack_word
                meta["answer"] = answer
                meta["choices"] = choices
                meta_info[question_id] = meta

            # Debug image generation
            make_debug = False
            if split == "val":
                # Original val behavior:
                #   - debug for first N_DEBUG_IMAGES by index, or
                #   - always debug for a specific DEBUG_IMAGE_ID
                if i < n_debug_images or (
                    debug_image_id is not None and image_id == debug_image_id
                ):
                    make_debug = True
            else:
                # Original train behavior: debug for first 5 samples
                if i < 5:
                    make_debug = True

            if make_debug:
                res = apply_pattern_for_image(
                    image=image,
                    ocr_info=ocr_info,
                    text_to_place=attack_word,
                    pattern=pat,
                    config=config,
                    image_id=image_id,
                    question_id=question_id,
                    subset_name=split,
                    answer_text=(
                        answer
                        if pat.get("placement", {}).get("mode", "")
                        == "grid3x3_bucketed_simple"
                        else None
                    ),
                    is_debug=True,
                )
                debug_img = res["image"]
                debug_meta = res["metadata"]
                debug_fname = f"{image_id}_{question_id}_{split}__debug.jpg"
                debug_image_dir = os.path.join(this_out_dir, "debug_images")
                os.makedirs(debug_image_dir, exist_ok=True)
                debug_out_path = os.path.join(debug_image_dir, debug_fname)
                debug_img.save(debug_out_path)
                print(f"Saved debug image to {debug_out_path}")

                if split == "val" and debug_image_id is not None and image_id == debug_image_id:
                    print(
                        f"Debug image ID {debug_image_id} shown, "
                        f"stopping further debug images for this choice_level."
                    )
                    is_debug_image_id_shown = True

            # Early break only for "val" when running in debug-only mode
            # and the special debug_image_id has been processed.
            if split == "val" and is_only_debug and is_debug_image_id_shown:
                break

        # Save metadata for this choice level
        if not is_only_debug:
            meta_json_path = os.path.join(
                this_out_dir, f"obj_attack_{choice_level}_meta.json"
            )
            with open(meta_json_path, "w") as f:
                json.dump(meta_info, f, indent=2)
            print(
                f"Saved metadata for {choice_level} obj-attack to {meta_json_path}"
            )

        # Reset flag for next choice_level (if any)
        is_debug_image_id_shown = False

    print(f"Finished obj-attack generation for split={split}.")


if __name__ == "__main__":
    BASE_DIR = "../data"

    # Global debug config (for val split; train keeps the original "i < 5" behavior)
    IS_ONLY_DEBUG = False
    N_DEBUG_IMAGES = 10  # per choice level (by index)
    DEBUG_IMAGE_ID = "85cbb7667b273c8d"

    # Settings for each split
    split_settings = [
        {
            "split": "val",
            "hf_split": "validation",
            "config_path": "typo_attack_config/obj_attack/val_config.json",
            "ocr_json_path": "assets/textvqa_meta/TextVQA_Rosetta_OCR_v0.2_val.json",
            "mc_json_path": "assets/mcq_data/textvqa_val_mcq_samples.json",
            "choice_levels": ["correct", "easy", "medium", "hard"],  # same as original val script
            "is_only_debug": IS_ONLY_DEBUG,
            "n_debug_images": N_DEBUG_IMAGES,
            "debug_image_id": DEBUG_IMAGE_ID,
        },
        {
            "split": "train",
            "hf_split": "train",
            "config_path": "typo_attack_config/obj_attack/train_config.json",
            "ocr_json_path": "assets/textvqa_meta/TextVQA_Rosetta_OCR_v0.2_train.json",
            "mc_json_path": "assets/mcq_data/textvqa_train_mcq_samples.json",
            "choice_levels": ["correct", "easy", "medium", "hard"],  # same as original train script
            "is_only_debug": False,      # train originally did not support debug-only mode
            "n_debug_images": 0,         # unused for train (uses i<5 rule)
            "debug_image_id": None,      # unused for train
        },
    ]

    for cfg in split_settings:
        generate_obj_attack_for_split(
            split=cfg["split"],
            hf_split=cfg["hf_split"],
            config_path=cfg["config_path"],
            ocr_json_path=cfg["ocr_json_path"],
            mc_json_path=cfg["mc_json_path"],
            choice_levels=cfg["choice_levels"],
            base_dir=BASE_DIR,
            is_only_debug=cfg["is_only_debug"],
            n_debug_images=cfg["n_debug_images"],
            debug_image_id=cfg["debug_image_id"],
        )

    print("All done.")
