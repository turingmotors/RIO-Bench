import re
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# ===========================
# Text normalization
# ===========================
def _norm(s: str) -> str:
    """Lowercase, trim, collapse inner spaces; keep only alnum+space for robust substring checks."""
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = " ".join(s.split())
    # keep alnum and spaces
    return re.sub(r"[^a-z0-9 ]+", "", s)

# ===========================
# Choice helpers
# ===========================
def _letters(n: int) -> List[str]:
    """['A','B',... ] of length n."""
    base = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return list(base[:n])

def _letter_to_idx(letter: str) -> Optional[int]:
    """Map 'A'/'a' -> 0, 'B' -> 1, ... ; return None if invalid."""
    if not isinstance(letter, str) or not letter:
        return None
    ch = letter.strip().upper()
    if len(ch) != 1:
        return None
    if "A" <= ch <= "Z":
        return ord(ch) - ord("A")
    return None

def _best_letter_from_text(resp: str, n_choices: int) -> Optional[int]:
    """
    Extract a choice letter from a free-form response.
    Prioritize explicit patterns like 'response: (c)', 'assistant: b', '(d)', 'c)', 'c.' or bare 'c'.
    Returns index (0-based) or None.
    """
    if not isinstance(resp, str):
        return None
    s = resp.strip()

    # Strong patterns (with keywords)
    pat_strongs = [
        r"(?:answer|assistant)\s*:\s*\(?([a-z])\)?",
        r"(?:the\s+answer\s+is|choose)\s*\(?([a-z])\)?",
    ]
    for pat in pat_strongs:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            idx = _letter_to_idx(m.group(1))
            if idx is not None and idx < n_choices:
                return idx

    # Medium patterns like "(b)", "b)", "b.", "[b]"
    pat_mediums = [
        r"\(([a-z])\)", r"\b([a-z])\)", r"\b([a-z])\.", r"\[([a-z])\]"
    ]
    for pat in pat_mediums:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            idx = _letter_to_idx(m.group(1))
            if idx is not None and idx < n_choices:
                return idx

    # Bare single-letter token (avoid catching letters inside words)
    # We scan in order A..Z so if multiple appear, the first valid wins.
    for L in _letters(n_choices):
        if re.search(rf"\b{L}\b", s, flags=re.IGNORECASE):
            return _letter_to_idx(L)

    return None

def _match_by_choice_text(resp: str, choices: List[str]) -> Optional[int]:
    """
    If the model outputs the option text (or a substring), try to align it to one choice.
    Strategy:
      - normalized substring match: choice in resp OR resp in choice
      - if multiple match, pick the longest choice match (more specific).
    """
    s = _norm(resp)
    if not s:
        return None
    cand = []
    for i, ch in enumerate(choices):
        c = _norm(ch)
        if not c:
            continue
        if c in s or s in c:
            cand.append((len(c), i))
    if not cand:
        return None
    # choose the most specific (longest normalized choice text)
    cand.sort(reverse=True)
    return cand[0][1]


# ===========================
# Main evaluator
# ===========================
def evaluate_multiple_choice(
    conversations: List[Any],
    responses: List[str],
    data: Dict[str, List[Any]],
    allow_text_match: bool = True,
    gt_key: str = "correct_letter",
    pred_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate multiple-choice VQA.

    gt_key: key of ground truth label (e.g., 'correct_letter', 'correct_idx', 'label', etc.)
    pred_key: key of predicted label (usually not needed if using responses)
    """
    # 先頭で全キーの長さチェック
    N = len(responses)
    for key in ["question_id", "image_id", "question", "choices", "answer", "correct_letter"]:
        if key in data and isinstance(data[key], list) and len(data[key]) != N:
            data[key] = data[key][:N]

    images     = data.get("image",   [""] * len(responses))
    questions  = data.get("question",[""] * len(responses))
    all_choices: List[List[str]] = data.get("choices", [[] for _ in responses])
    if isinstance(all_choices[0], dict):
        all_choices = [list(c.values()) for c in all_choices]

    correct_total = 0
    records = []
    per_option = {}

    for i in tqdm(range(len(responses)), desc="Evaluating MC"):
        conv   = conversations[i] if i < len(conversations) else ""
        resp    = responses[i] if pred_key is None else data[pred_key][i]
        q      = questions[i] if i < len(questions) else ""
        chs    = all_choices[i] if i < len(all_choices) else []
        nC     = len(chs)

        # Resolve answer index
        item = {k: (v[i] if isinstance(v, list) and len(v) == len(responses) else None) for k, v in data.items()}
        answer_idx = None
        if isinstance(item[gt_key], int):
            answer_idx = item[gt_key]
        elif isinstance(item[gt_key], str):
            answer_idx = _letter_to_idx(item[gt_key])
            if answer_idx is None and allow_text_match and nC > 0:
                answer_idx = _match_by_choice_text(item[gt_key], chs)
        else:
            print(item)
            print(resp, chs)
            raise ValueError(f"Unsupported gt_key type: {type(item[gt_key])} for key '{gt_key}'")
        # print(item, answer_idx, resp, chs)

        # Predict index from response
        pred_idx = None
        if nC > 0:
            pred_idx = _best_letter_from_text(resp, nC)
            if pred_idx is None and allow_text_match:
                pred_idx = _match_by_choice_text(resp, chs)

        is_correct = int(pred_idx is not None and answer_idx is not None and pred_idx == answer_idx)
        correct_total += is_correct


        rec = {
            "image_id": data["image_id"][i] if "image_id" in data and i < len(data["image_id"]) else None,
            "question_id": data["question_id"][i] if "question_id" in data and i < len(data["question_id"]) else None,
            "question": q,
            "conversation": conv,
            "choices": chs,
            "answer_idx": answer_idx,
            "answer": chs[answer_idx] if (answer_idx is not None and nC > 0) else None,
            "response": resp,
            "pred_idx": pred_idx,
            "pred_letter": _letters(nC)[pred_idx] if (pred_idx is not None and nC > 0) else None,
            "is_correct": is_correct,
        }
        records.append(rec)

    accuracy = correct_total / len(responses) if responses else 0.0

    print(f"Accuracy: {accuracy:.4f}")
    return {"accuracy": accuracy, "records": records} 


if __name__ == "__main__":
    # data の例
    data = {
        "image": [
            "img1.jpg",
            "img2.jpg",
            "img3.jpg"
        ],
        "question": [
            "What is this object?", 
            "Which animal is shown?",
            "What is the color of the sky?"
        ],
        "choices": [
            ["Cassette deck", "Musical keyboard", "Stairs", "Crown"],   # A,B,C,D
            ["Cat", "Dog", "Koala", "Parrot"],                        # A,B,C,D
            ["Blue", "Green", "Red", "Yellow"]                        # A,B,C,D

        ],
        "correct_letter": ["A", "C", "A"]  # 正解ラベル
    }

    # conversations: model prompts/history (for logging)
    conversations = ["...prompt for #1...", "...prompt for #2...", "...prompt for #3..."]

    # responses: model outputs (free text allowed)
    responses = [
        "I think the answer is (A).",
        "The correct choice is Koala.",  # Text match is also acceptable
        "A."   
    ]

    out = evaluate_multiple_choice(conversations, responses, data, gt_key="correct_letter", )
    print(out["accuracy"])