from typing import Dict, List, Optional
import random
import re

def _normalize_short_answer(s: str) -> str:
    """Lowercase, strip punctuation/spaces for stable supervision."""
    s = s.strip().lower()
    s = re.sub(r"[^\w\s:./-]", "", s)   # keep common textvqa symbols minimally
    s = re.sub(r"\s+", " ", s)
    return s

def _answer2score_to_dict(answer2score):
    if isinstance(answer2score, dict):
        return answer2score
    if isinstance(answer2score, list):
        out = {}
        for item in answer2score:
            if isinstance(item, dict):
                ans = item.get("answer")
                score = item.get("score")
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                ans, score = item
            else:
                continue
            if ans is not None:
                out[ans] = score
        return out
    return {}

def _pick_open_answer(example: Dict) -> str:
    """
    Pick a single supervision string from answers/answer2score.
    Priority:
      1) answer2score top (if available, highest score non-null)
      2) majority vote in answers
      3) first answers[0]
    """
    answers = example.get("answers", []) or []
    a2s = _answer2score_to_dict(example.get("answer2score", None))
    if a2s:
        # choose key with max score (ignore None)
        scored = [(k, v) for k, v in a2s.items() if isinstance(v, (int, float))]
        if scored:
            scored.sort(key=lambda kv: kv[1], reverse=True)
            return _normalize_short_answer(scored[0][0])

    # fallback
    return ""

def _majority_vote(answers: List[str]) -> str:
    """Return the most common answer from a list of answers."""
    if not answers:
        return "unknown"
    norm = [_normalize_short_answer(a) for a in answers]
    best = max(set(norm), key=norm.count)
    return best

def _combine_answers_with_commas(answers: List[str]) -> str:
    """
    Combine all unique answers into a single string, separated by commas.
    """
    unique_answers = list(set(_normalize_short_answer(a) for a in answers if a.strip()))
    return ", ".join(unique_answers) if unique_answers else "unknown"

def _llava_eos(model_name: str) -> str:
    return "</s>" if "llava" in (model_name or "").lower() else ""

def _wrap_messages(first_image, turns, eos: str, peft_ver: bool):
    """
    turns: list of dicts like {"from": "human"/"gpt", "value": "text"}
    Build messages without any task identifiers.
    """
    messages = []
    for i, conv in enumerate(turns):
        if conv["from"] == "human":
            user_content = []
            if i == 0 and first_image is not None:
                user_content.append({"type": "image", "image": first_image})
            user_content.append({"type": "text", "text": conv["value"]})
            messages.append({"role": "user", "content": user_content})
        elif conv["from"] == "gpt":
            messages.append({"role": "assistant", "content": [{"type": "text", "text": conv["value"] + eos}]})
        else:
            raise ValueError(f"Unsupported role: {conv['from']}")

    if peft_ver:
        return [{"images": [first_image], "messages": messages}]
    else:
        return [{"messages": messages}]

# -----------------------------
# Obj-MCQ
# -----------------------------
def prepare_data_obj_mcq(example: Dict, model_name: str, peft_ver: bool = False):
    """
    Expect fields:
      - image (PIL)
      - question (str)  # already includes (A)...(D) and 'Answer with only the option letter...'
      - choices (dict: {"A": "...", ...})
      - answer (str) in {"A","B","C","D"}
    No task identifiers; instruction lives in natural language inside the question.
    """
    eos = _llava_eos(model_name)
    image = example["image"]
    q = example["question"]
    gold_letter = example["answer"].strip()

    # human: plain question; assistant: letter only
    turns = [
        {"from": "human", "value": q},
        {"from": "gpt",   "value": gold_letter}
    ]
    return _wrap_messages(image, turns, eos=eos, peft_ver=peft_ver)

# -----------------------------
# Obj-OpenEnded
# -----------------------------
def prepare_data_obj_open(example: Dict, model_name: str, peft_ver: bool = False):
    """
    Expect fields:
      - image (PIL)
      - question (str)  # e.g., 'What objects can be seen in the image?'
      - answers (list[str]) and optionally answer2score (dict)
    Output: concise answer in one sentence.
    """
    eos = _llava_eos(model_name)
    image = example["image"]
    q = example["question"]
    y = _combine_answers_with_commas(example.get("answers", [])) # Use all unique answers for training.

    turns = [
        {"from": "human", "value": q},
        {"from": "gpt",   "value": y}
    ]
    return _wrap_messages(image, turns, eos=eos, peft_ver=peft_ver)

# -----------------------------
# Text-OpenEnded
# -----------------------------
def prepare_data_text_open(example: Dict, model_name: str, peft_ver: bool = False):
    """
    Expect fields (TextVQA-like):
      - image (PIL)
      - question (str)
      - answers (list[str]) (10-way annotation)
    We don't reveal 'must read text' tag. Keep the same neutral format hint.
    """
    eos = _llava_eos(model_name)
    image = example.get("image")
    q = example["question"]
    y = _majority_vote(example.get("answers", [])) # As default, pick most agreed answer.

    turns = [
        {"from": "human", "value": q},
        {"from": "gpt",   "value": y}
    ]
    return _wrap_messages(image, turns, eos=eos, peft_ver=peft_ver)


def prepare_data_dispatch(split_name: str, subset_name: str, example: Dict, model_name: str, peft_ver: bool = False):
    """
    split_name: one of {"obj_attack","obj_clean","txt_attack","text_attack","text_clean", ...}
    subset_name: one of {"mcq_*","open_ended_*", ...} — your directory granularity

    Returns messages_list: [{"messages": [...]}] or [{"images":[...], "messages":[...]}] if peft_ver=True
    """
    s = split_name.lower()
    t = subset_name.lower()

    if s.startswith("obj") and t.startswith("mcq"):
        return prepare_data_obj_mcq(example, model_name, peft_ver)
    elif s.startswith("obj") and "open" in t:
        return prepare_data_obj_open(example, model_name, peft_ver)
    elif s.startswith(("txt", "text")) and "open" in t:
        return prepare_data_text_open(example, model_name, peft_ver)
    else:
        # Fallback to open-ended style for unknown subsets
        return prepare_data_obj_open(example, model_name, peft_ver)


def preprocess_dataset_train(
        dataset, 
        split_name: str, 
        subset_name: str, 
        model_name: str, 
        peft_ver: bool = False
    ):
    """
    dataset: a HuggingFace Dataset object
    split_name: one of {"obj_attack","obj_clean","txt_attack","text_attack","text_clean", ...}
    subset_name: one of {"mcq_*","open_ended_*", ...} — your directory granularity
    model_name: e.g., "LLaVA-1.5-7B", "mistral-7b-instruct-v0.1", etc.
    peft_ver: if True, include images in the output dicts

    Returns a list of {"messages": [...]} or {"images":[...], "messages":[...]} dicts.
    """
    processed = []
    for i, example in enumerate(dataset):
        try:
            msgs_list = prepare_data_dispatch(split_name, subset_name, example, model_name, peft_ver)
            processed.extend(msgs_list)
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            continue
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} examples...")
    return processed


if __name__ == "__main__":
    import os
    import glob
    from datasets import load_from_disk, DatasetDict, Dataset
    import argparse
    import json

    data_dir = "../data/RIO-Bench/hf_dataset_unique_img"
    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    peft_ver = False  # if True, output dicts include "images" key

    split = "train"  # or "val"
    ds_num_each = 100  # for quick testing; set to None to process all

    processed_datasets = []
    for t in ["obj_attack", "txt_attack"]:
        dataset_dirs = glob.glob(os.path.join(data_dir, split, t, "*"))
        for _d in dataset_dirs:
            print(f"Processing {_d} ...")
            ds = load_from_disk(_d)
            print(f"  Original size: {len(ds)}")
            if ds_num_each is not None:
                ds = ds.shuffle(seed=42)
                ds = ds.select(range(min(ds_num_each, len(ds))))
            print(f"  Selected size: {len(ds)}")
            processed_msgs = preprocess_dataset_train(ds, split_name=t, subset_name=os.path.basename(_d), model_name=model_name, peft_ver=peft_ver)
            print(f"  Processed size: {len(processed_msgs)}")
            print(f"  Sample message: {processed_msgs[0]}")
            processed_datasets.extend(processed_msgs)
            print()
            del ds  # free memory
    print(f"Total processed examples: {len(processed_datasets)}")
    from itertools import chain
    processed_train_dataset = list(chain.from_iterable(processed_datasets))
    print(f"Total flattened processed examples: {len(processed_train_dataset)}")
