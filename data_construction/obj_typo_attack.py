from typo_attack_utils import apply_pattern_for_image
from typing import List, Dict, Tuple, Set, Union, Iterable
from PIL import Image
import PIL
import random
import os

DEFAULT_GRID_ORDER = ("C","TL","TR","BL","BR","TC","BC","CL","CR")

if __name__ == "__main__":
    from datasets import load_dataset
    import json
    import random

    IS_ONLY_DEBUG = False
    N_DEBUG_IMAGES = 10  # per choice level
    DEBUG_IMAGE_ID = "85cbb7667b273c8d"
    is_debug_image_id_shown = False

    # -----------------------------
    # Validation set obj-attack images
    # -----------------------------
    split = "val"
    print(f"Generating obj-attack images for TextVQA {split} set")

    # config
    config_path = "typo_attack_config/obj_attack/val_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    print(f"Loaded config from {config_path}")
    pat = config["patterns"][0]

    # load TextVQA validation set
    ds = load_dataset("facebook/textvqa", split="validation")

    # load OCR data
    textvqa_ocr_json = "assets/textvqa/TextVQA_Rosetta_OCR_v0.2_val.json"
    with open(textvqa_ocr_json, "r") as f:
        textocr_data = json.load(f)
    print(f"Loaded {textvqa_ocr_json}")

    # load multichoice data
    mc_json = "textvqa_val_mcq_samples.json"
    with open(mc_json, "r") as f:
        mc_data = json.load(f)
    mc_data_dict = {item["question_id"]: item for item in mc_data}

    image_id2ocr = {}
    for item in textocr_data["data"]:
        image_id2ocr[item["image_id"]] = item


    BASE_DIR = "../data/RIO-Bench"
    OUT_DIR = os.path.join(BASE_DIR, split, "obj_attack")
    os.makedirs(OUT_DIR, exist_ok=True)

    random.seed(42)  # for reproducibility

    # for choice_level in ["easy", "medium", "hard"]:
    for choice_level in ["correct"]:
        print(f"Generating obj-attack images for choice level: {choice_level}")
        THIS_OUT_DIR = os.path.join(OUT_DIR, pat["pattern_id"], choice_level)
        os.makedirs(THIS_OUT_DIR, exist_ok=True)

        THIS_IMAGE_DIR = os.path.join(THIS_OUT_DIR, "images")
        os.makedirs(THIS_IMAGE_DIR, exist_ok=True)

        meta_info = {}

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

            # obj-attack
            if not IS_ONLY_DEBUG:
                res = apply_pattern_for_image(
                    image=image,
                    ocr_info=ocr_info,
                    text_to_place=attack_word,
                    pattern=pat,
                    config=config,
                    image_id=image_id,
                    question_id=question_id,
                    subset_name=split,
                    answer_text=answer if pat.get("placement", {}).get("mode","") == "grid3x3_bucketed_simple" else None,
                )
                
                out_img = res["image"]
                meta = res["metadata"]

                fname = f"{image_id}_{question_id}_{split}__{meta['pattern_id']}__seed{meta['seed']}.jpg"
                out_path = os.path.join(THIS_IMAGE_DIR, fname)
                out_img.save(out_path)
                # save meta info
                meta["attack_word"] = attack_word
                meta["answer"] = answer
                meta["choices"] = choices
                meta_info[question_id] = meta

            if i < N_DEBUG_IMAGES or image_id == DEBUG_IMAGE_ID:
                res = apply_pattern_for_image(
                    image=image,
                    ocr_info=ocr_info,
                    text_to_place=attack_word,
                    pattern=pat,
                    config=config,
                    image_id=image_id,
                    question_id=question_id,
                    subset_name=split,
                    answer_text=answer if pat.get("placement", {}).get("mode","") == "grid3x3_bucketed_simple" else None,
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

                if image_id == DEBUG_IMAGE_ID:
                    print(f"Debug image ID {DEBUG_IMAGE_ID} shown, stopping further debug images.")
                    is_debug_image_id_shown = True

            # if i == N_DEBUG_IMAGES - 1 and IS_ONLY_DEBUG:
            #     break
            if is_debug_image_id_shown and IS_ONLY_DEBUG:
                break

        # save meta info
        if not IS_ONLY_DEBUG:
            meta_json_path = os.path.join(THIS_OUT_DIR, f"obj_attack_{choice_level}_meta.json")
            with open(meta_json_path, "w") as f:
                json.dump(meta_info, f, indent=2)
            print(f"Saved metadata for {choice_level} obj-attack to {meta_json_path}")

        is_debug_image_id_shown = False

    print("All done.")

    exit()

    # -----------------------------
    # Train set obj-attack images
    # -----------------------------
    split = "train"
    print(f"Generating obj-attack images for TextVQA {split} set")
    # (similar to above, but for train set and without multichoice filtering)

    # config
    config_path = "typo_attack_config/obj_attack/train_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    print(f"Loaded config from {config_path}")

    pat = config["patterns"][0]
    
    # load TextVQA train set
    ds = load_dataset("facebook/textvqa", split=split)

    # load OCR data
    textvqa_ocr_json = "assets/textvqa/TextVQA_Rosetta_OCR_v0.2_train.json"
    with open(textvqa_ocr_json, "r") as f:
        textocr_data = json.load(f)
    print(f"Loaded {textvqa_ocr_json}")

    # load multichoice data
    mc_json = f"textvqa_{split}_mcq_samples.json"
    with open(mc_json, "r") as f:
        mc_data = json.load(f)
    mc_data_dict = {item["question_id"]: item for item in mc_data}

    image_id2ocr = {}
    for item in textocr_data["data"]:
        image_id2ocr[item["image_id"]] = item


    BASE_DIR = "../data/RIO-Bench"
    OUT_DIR = os.path.join(BASE_DIR, split, "obj_attack")
    os.makedirs(OUT_DIR, exist_ok=True)

    random.seed(42)  # for reproducibility

    for choice_level in ["easy", "medium", "hard"]:
        print(f"Generating obj-attack images for choice level: {choice_level}")
        THIS_OUT_DIR = os.path.join(OUT_DIR, pat["pattern_id"], choice_level)
        os.makedirs(THIS_OUT_DIR, exist_ok=True)

        THIS_IMAGE_DIR = os.path.join(THIS_OUT_DIR, "images")
        os.makedirs(THIS_IMAGE_DIR, exist_ok=True)

        meta_info = {}

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

            if choice_level == "hard":
                attack_word = c_hard
            elif choice_level == "medium":
                attack_word = c_medium
            else:  # easy
                attack_word = c_easy

            # obj-attack
            res = apply_pattern_for_image(
                image=image,
                ocr_info=ocr_info,
                text_to_place=attack_word,
                pattern=pat,
                config=config,
                image_id=image_id,
                question_id=question_id,
                subset_name=split,
                answer_text=answer if pat.get("placement", {}).get("mode","") == "grid3x3_bucketed_simple" else None,
            )
            
            out_img = res["image"]
            meta = res["metadata"]

            fname = f"{image_id}_{question_id}_{split}__{meta['pattern_id']}__seed{meta['seed']}.jpg"
            out_path = os.path.join(THIS_IMAGE_DIR, fname)
            out_img.save(out_path)
            # save meta info
            meta["attack_word"] = attack_word
            meta["answer"] = answer
            meta["choices"] = choices
            meta_info[question_id] = meta

            if i < 5:
                res = apply_pattern_for_image(
                    image=image,
                    ocr_info=ocr_info,
                    text_to_place=attack_word,
                    pattern=pat,
                    config=config,
                    image_id=image_id,
                    question_id=question_id,
                    subset_name=split,
                    answer_text=answer if pat.get("placement", {}).get("mode","") == "grid3x3_bucketed_simple" else None,
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
                # if i == 4:
                #     break

        # save meta info
        meta_json_path = os.path.join(THIS_OUT_DIR, f"obj_attack_{choice_level}_meta.json")
        with open(meta_json_path, "w") as f:
            json.dump(meta_info, f, indent=2)
        print(f"Saved metadata for {choice_level} obj-attack to {meta_json_path}")

    print("All done.")