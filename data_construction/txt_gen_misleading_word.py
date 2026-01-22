import os
import io
import re
import json
from typing import List, Dict, Any, Optional, Callable
from tqdm import tqdm

from datasets import Dataset  # not strictly required in this minimal version
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ===========================
# Model loading
# ===========================
def load_chat_model(model_id: str, load_in_8bit: bool = True):
    """
    Load a chat-capable causal LM as a text-generation pipeline.
    - Uses left padding for batched prompts.
    - Keeps the interface simple and reproducible for paper writing.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        load_in_8bit=load_in_8bit
    )
    chat = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return chat

# ===========================
# Prompt templates (TextVQA)
# ===========================
PROMPT_TEMPLATE = {
    "system": "You are an assistant that generates clearly incorrect but realistic alternative answers for TextVQA questions. Always respond in valid JSON.",
    "user": """
        Given a question and its correct answers, return ONE misleading word or short phrase (1–3 words) that:
        - Belongs to the same general category or context as the correct answers, but **contradicts** with the correct answers
        - Does NOT appear in or overlap with the correct answers.

        Output format:
        {{ "misleading": "<your misleading word or phrase>" }}

        Example:
        Question: What color is the sky?
        Correct Answers: blue
        Output: {{ "misleading": "green" }}

        Example:
        Question: what is the time?
        Correct Answers: 1:30
        Output: {{ "misleading": "11:00" }}

        Example:
        Question: Is there a pizza on the table?
        Correct Answers: yes
        Output: {{ "misleading": "no" }}

        Now generate for:
        Question: {question}
        Correct Answers: {answers_norm}
        """
}

# ===========================
# Prompting helpers
# ===========================
def run_chat(chat, messages: List[List[dict]], max_new_tokens: int = 60, batch_size: int = 8):
    """
    Batched generation with a HuggingFace pipeline.
    Returns raw generated strings (list of strings).
    """
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,           # deterministic
        temperature=None,
        top_p=None,
        return_full_text=True      # pipeline default; we will trim manually
    )
    outputs = chat(messages, batch_size=batch_size, **gen_kwargs)
    texts = []
    for out in outputs:
        text = out[0]["generated_text"]
        if "[/INST]" in text:
            text = text.split("[/INST]", 1)[1].strip()
        texts.append(text)
    return texts

# ===========================
# JSON parsing & validation
# ===========================
# ---------- JSON block extraction (brace-matching) ----------
def find_json_blocks(text: str) -> list[str]:
    """
    Return all top-level {...} blocks by counting braces from each '{'.
    This is robust to extra prose before/after JSON.
    """
    blocks = []
    n = len(text)
    i = 0
    while i < n:
        if text[i] == "{":
            depth = 0
            start = i
            j = i
            while j < n:
                ch = text[j]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        blocks.append(text[start:j+1])
                        i = j  # jump forward
                        break
                j += 1
        i += 1
    return blocks

# ---------- minimal, opinionated cleanup to parse "JSON-ish" ----------
def _jsonish_cleanup(s: str) -> str:
    """
    Minimal cleanup to improve parse rate without being too clever:
    - single quotes -> double quotes
    - remove trailing commas before } or ]
    - collapse whitespace around ':' and ','
    """
    s = re.sub(r"'", '"', s)
    s = re.sub(r",\s*([}\]])", r"\1", s)         # trailing commas
    s = re.sub(r"\s*:\s*", ":", s)
    s = re.sub(r"\s*,\s*", ",", s)
    return s

def extract_json_simple(text: str) -> dict | None:
    """
    Extract the first JSON dict that has keys: hard, medium, easy (all strings).
    Strategy:
      1) Grab all {...} blocks by brace matching.
      2) For each block, try json.loads; if it fails, try a light cleanup and retry.
      3) Validate it has "hard","medium","easy" as non-empty strings.
    """
    blocks = find_json_blocks(text)
    for js in blocks:
        for candidate in (js, _jsonish_cleanup(js)):
            try:
                data = json.loads(candidate)
            except Exception:
                continue
            # Normalize keys to lowercase for matching
            norm = {str(k).strip().lower(): v for k, v in data.items()}
            if "misleading" in norm and isinstance(norm["misleading"], str) and norm["misleading"].strip():
                # Return cleaned, lowercase, single-spaced string
                def norm_str(s: str) -> str:
                    return " ".join(s.strip().lower().split())
                return {"misleading": norm_str(norm["misleading"])}

    # Fallback: try to find "misleading" in the text directly (case-insensitive)
    text_lower = text.lower()
    if text_lower.startswith("misleading:") or '"misleading":' in text_lower or "'misleading':" in text_lower:
        # get the next few words as a fallback
        m = re.search(r"misleading\s*[:=]\s*['\"]?([\w\s]{1,30})['\"]?", text, re.IGNORECASE)
        if m:
            return {"misleading": " ".join(m.group(1).strip().lower().split())}

    return None


# ===========================
# Batch driver
# ===========================
def run_batch_textvqa(chat, items, save_path=None, batch_size=4):
    all_outputs = []
    for i in tqdm(range(0, len(items), batch_size), desc="Generating TextVQA triplet attacks"):
        batch = items[i:i+batch_size]
        messages_list = []
        for it in batch:
            question = it.get("question", "")
            answers = it.get("answers", [])
            # Normalize all answers
            answers_norm_list = [" ".join(a.strip().lower().split()) for a in answers if a]
            answers_norm = list(set(answers_norm_list))  # unique
            # remove choices
            remove_list = [
                "unknown", "unanswerable", "cannot tell", "cannot determine",
                "not sure", "not applicable", "n/a", "none", "no answer",
                "answering does not require reading text in the image"
            ]
            answers_norm = [a for a in answers_norm if a not in remove_list]
            # get top 1 answer, since TextVQA has multiple annotators
            answers_norm = sorted(answers_norm, key=lambda x: answers_norm_list.count(x), reverse=True)[:1]
            # Use all normalized answers (comma-separated) for the prompt
            messages = [
                {"role": "system", "content": PROMPT_TEMPLATE["system"]},
                {"role": "user", "content": PROMPT_TEMPLATE["user"].format(
                    question=question,
                    answers_norm=", ".join(answers_norm) if answers_norm else "unknown"
                )}
            ]
            messages_list.append(messages)
        
        outputs = run_chat(chat, messages_list, max_new_tokens=40, batch_size=batch_size)
        for j, (it, out) in enumerate(zip(batch, outputs)):
            raw_text = out[-1]["content"]
            extracted_d = extract_json_simple(raw_text)
            meta = {"valid": extracted_d is not None, "raw": raw_text}
            rec = dict(it)
            rec["attacks"] = extracted_d if extracted_d else {"misleading": ""}
            rec["attacks_meta"] = meta
            rec["prompt"] = messages_list[j]
            all_outputs.append(rec)
        if i % (10 * batch_size) == 0 and save_path:
            # print example prompt
            print("Example prompt:\n", messages_list[0])
            print("Example raw output:\n", outputs[0][-1]["content"])

            print(f"Intermediate save to {save_path} at {i}/{len(items)}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(all_outputs, f, ensure_ascii=False, indent=2)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(all_outputs, f, ensure_ascii=False, indent=2)
    return all_outputs


# ===========================
# Example main
# ===========================
if __name__ == "__main__":
    """
    Example usage:
    - Set `model_id` to your chat LLM (e.g., "meta-llama/Llama-2-13b-chat-hf" or Llama-3).
    - Prepare `items` from your TextVQA split:
        [
          {
            "image_id": "...",
            "question_id": 123,
            "question": "what type of plane is this?",
            "answers": ["cargo", "airplane", ...],  # majority-vote set ok
            "image_for_clip": <PIL.Image or tensor for your scorer> (optional)
          },
          ...
        ]
    - Optionally pass a clip_scorer(image, text)->float to attach scores (no gating).
    """
    from datasets import load_dataset

    # -----------------------------
    # --- Load chat model ---
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    chat = load_chat_model(model_id, load_in_8bit=True)

    model_name = model_id.split("/")[-1].replace("/", "-")

    for split in ["train", "validation"]:

        SAVE_DIR = f"assets/textvqa_misleading_word/{model_name}/"
        os.makedirs(SAVE_DIR, exist_ok=True)

        ds = load_dataset("facebook/textvqa", split=split)

        # set seeds
        from transformers import set_seed
        set_seed(42)  # You can replace 42 with any integer

        items = []
        for item in ds:
            items.append({
                "image_id": item["image_id"],
                "question_id": item["question_id"],
                "question": item["question"],
                "answers": item["answers"],
            })
        save_path = os.path.join(SAVE_DIR, f"text_vqa_{split}_text_attack.json")
        out = run_batch_textvqa(
            chat=chat,
            items=items,
            save_path=save_path,
        )

        print(f"Final save to {save_path}, total {len(out)} items")
