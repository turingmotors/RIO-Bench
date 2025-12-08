# text_placer.py
import PIL
from PIL import Image, ImageDraw, ImageFont, ImageStat
from typing import List, Dict, Tuple, Optional, Iterable, Any
import math
import os
from pathlib import Path
import json
import random
import hashlib
import sys
import unicodedata
import re

Rect = Tuple[int, int, int, int]  # (x0, y0, x1, y1)
GridCode = Tuple[str, ...]
DEFAULT_GRID_ORDER: GridCode = ("C","TL","TR","BL","BR","TC","BC","CL","CR")

FIXED_FONT = str(Path("assets") / "fonts" / "dejavu-sans-ttf-2.37/ttf/DejaVuSans.ttf")
print("Using fixed font:", FIXED_FONT)

_num_re = re.compile(r"\d+(?:[.,]\d+)*")  # 10, 10.5, 1,000.25 (NFKC handled below)
_ws = re.compile(r"\s+")
_punct = re.compile(r"[^\w]+", flags=re.UNICODE)

# -----------------------------
# Basic geometry utilities
# -----------------------------
def _rect_is_valid(r: Optional[Rect]) -> bool:
    return (r is not None) and (r[2] > r[0]) and (r[3] > r[1])

def _to_pixel_rect(bbox_norm: Dict[str, float], W: int, H: int, pad_px: int = 0) -> Rect:
    x0 = int(round(bbox_norm["top_left_x"] * W))
    y0 = int(round(bbox_norm["top_left_y"] * H))
    w  = int(round(bbox_norm["width"] * W))
    h  = int(round(bbox_norm["height"] * H))
    x1 = x0 + w
    y1 = y0 + h
    x0 -= pad_px; y0 -= pad_px; x1 += pad_px; y1 += pad_px
    return (x0, y0, x1, y1)

def _clip_rect(r: Rect, W: int, H: int) -> Optional[Rect]:
    x0, y0, x1, y1 = r
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(W, x1); y1 = min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)

def _intersects(a: Rect, b: Rect) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)

def _to_pixel_rect_from_norm(bb_norm: Dict[str, float], W: int, H: int) -> Rect:
    x0 = int(round(bb_norm["top_left_x"] * W))
    y0 = int(round(bb_norm["top_left_y"] * H))
    w  = int(round(bb_norm["width"] * W))
    h  = int(round(bb_norm["height"] * H))
    return (x0, y0, x0 + w, y0 + h)

def _safe_textbbox(draw: ImageDraw.ImageDraw, xy, text: str, font,
                   stroke_width: int = 0, stroke_fill=None) -> Rect:
    """
    Call textbbox safely across Pillow versions.
    Some versions don't accept stroke_fill; fall back progressively.
    """
    try:
        l, t, r, b = draw.textbbox(xy, text, font=font,
                                   stroke_width=stroke_width,
                                   stroke_fill=(stroke_fill if stroke_width > 0 else None))
    except TypeError:
        try:
            l, t, r, b = draw.textbbox(xy, text, font=font,
                                       stroke_width=stroke_width)
        except TypeError:
            l, t, r, b = draw.textbbox(xy, text, font=font)
    return (l, t, r, b)

# -----------------------------
# OCR-derived forbidden areas
# -----------------------------
def prepare_forbidden_rects(ocr_info: List[Dict], image_size, pad_rel: float = 0.01) -> List[Rect]:
    W, H = image_size
    pad_px = int(round(pad_rel * max(W, H)))
    rects: List[Rect] = []
    for item in ocr_info:
        bb = item.get("bounding_box", {})
        if not {"top_left_x","top_left_y","width","height"} <= bb.keys():
            continue
        r = _to_pixel_rect(bb, W, H, pad_px=pad_px)
        cr = _clip_rect(r, W, H)
        if cr is not None:
            rects.append(cr)
    return rects

def _fits(rect: Rect, W: int, H: int, margin_px: int, forbidden: List[Rect]) -> bool:
    """Check margins and intersection with forbidden OCR rectangles."""
    x0, y0, x1, y1 = rect
    if x0 < margin_px or y0 < margin_px or x1 > W - margin_px or y1 > H - margin_px:
        return False
    for fr in forbidden:
        if _intersects(rect, fr):
            return False
    return True

# -----------------------------
# Font handling (ensure TTF so font_size works)
# -----------------------------
def _load_font(font_path: Optional[str], font_size: int) -> ImageFont.FreeTypeFont:
    # For reproducibility, ignore user font and use fixed DejaVuSans
    return ImageFont.truetype(FIXED_FONT, font_size)

def _text_rect(draw: ImageDraw.ImageDraw, xy, text: str, font, stroke_width: int = 0, stroke_fill=None) -> Rect:
    l, t, r, b = _safe_textbbox(draw, xy, text, font,
                                stroke_width=stroke_width,
                                stroke_fill=(stroke_fill if stroke_width > 0 else None))
    return (l, t, r, b)

# -----------------------------
# TextVQA: locate answer region from OCR
# -----------------------------
_Q_STOPWORDS = {
    # common function words; extend as needed
    "a","an","the","and","or","but","of","to","in","on","for","by","with","at","from","as",
    "is","are","was","were","be","being","been","do","does","did","have","has","had",
    "this","that","these","those","it","its","he","she","they","them","his","her","their",
    "there","here","over","under","between","about","which","what","who","whom","whose",
    "into","out","up","down","if","then","than","so","because","while","during","through",
    "can","could","should","would","may","might","will","shall","not","no","yes","true","false"
}

def find_support_rects_from_question(
    ocr_info: List[Dict],
    image_size: Tuple[int, int],
    question_text: str,
    *,
    min_chars: int = 3,
    per_token_min_score: float = 0.75,
    allow_numeric: bool = True,
    numeric_min_len: int = 2,
    max_keywords: int = 6
) -> List[Dict[str, Any]]:
    """Heuristic to find supporting OCR rects guided by question tokens."""
    W, H = image_size
    toks = [d.get("word", "") for d in ocr_info]
    boxes_norm = [d.get("bounding_box", {}) for d in ocr_info]
    boxes: List[Optional[Rect]] = []
    for bb in boxes_norm:
        if {"top_left_x", "top_left_y", "width", "height"} <= bb.keys():
            x0 = int(round(bb["top_left_x"] * W))
            y0 = int(round(bb["top_left_y"] * H))
            w = int(round(bb["width"] * W))
            h = int(round(bb["height"] * H))
            boxes.append((x0, y0, x0 + w, y0 + h))
        else:
            boxes.append(None)

    norm_toks = [_norm_robust(t) for t in toks]
    q_keys = _split_answer_keywords(question_text, min_token_chars=min_chars)[:max_keywords]
    results: List[Dict[str, Any]] = []

    # exact
    for i, nt in enumerate(norm_toks):
        if boxes[i] is None:
            continue
        if any(nt == _norm_robust(k) for k in q_keys):
            results.append(_pack_match(i, toks[i], nt, boxes[i], why="q_exact", score=1.0))

    # numeric
    if allow_numeric:
        q_nums = set([n for k in q_keys for n in _extract_numbers(k) if len(n.replace(".", "")) >= numeric_min_len])
        if q_nums:
            for i, raw in enumerate(toks):
                if boxes[i] is None:
                    continue
                cand_nums = [x for x in _extract_numbers(raw) if len(x.replace(".", "")) >= numeric_min_len]
                if set(cand_nums) & q_nums:
                    results.append(_pack_match(i, toks[i], norm_toks[i], boxes[i], why="q_numeric", score=1.0))

    # fuzzy
    have = {nt for nt in norm_toks}
    still = [_norm_robust(k) for k in q_keys if _norm_robust(k) not in have]
    for kw in still:
        best_i, best_sc = None, -1.0
        for i, nt in enumerate(norm_toks):
            if boxes[i] is None:
                continue
            sc = _token_sim(kw, nt)
            if sc > best_sc:
                best_sc, best_i = sc, i
        if best_i is not None and best_sc >= per_token_min_score:
            results.append(_pack_match(best_i, toks[best_i], norm_toks[best_i], boxes[best_i], why="q_fuzzy", score=best_sc))

    # deduplicate by rect
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for m in results:
        key = tuple(m["rect"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(m)
    return deduped

def _to_ascii_digits(s: str) -> str:
    # Normalize unicode widths and digits (e.g., '１０' -> '10')
    return unicodedata.normalize("NFKC", s)

def _extract_numbers(s: str) -> List[str]:
    s = _to_ascii_digits(s)
    nums = _num_re.findall(s)
    return [n.replace(",", "") for n in nums]

def _split_answer_keywords(answer: str, min_token_chars: int = 3) -> List[str]:
    # Split on whitespace/punct, normalize, drop short tokens
    norm = unicodedata.normalize("NFKC", answer).lower()
    rough = re.split(r"[\s/:\-_,.;!?()\[\]{}]+", norm)
    toks = []
    for t in rough:
        t = t.strip()
        if not t: 
            continue
        t = _punct.sub("", t)
        if len(t) >= min_token_chars:
            toks.append(t)
    return toks

def _norm_robust(s: str) -> str:
    # Unicode-compatible normalization, lowercase, strip punctuation and spaces
    s = unicodedata.normalize("NFKC", s).lower().strip()
    s = _ws.sub(" ", s)
    s = _punct.sub("", s)
    return s

def _levenshtein(a: str, b: str) -> int:
    # Edit distance (O(len(a)*len(b)))
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0]*len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1,        # deletion
                         cur[j-1] + 1,       # insertion
                         prev[j-1] + cost)   # substitution
        prev = cur
    return prev[-1]

def _lcs_len(a: str, b: str) -> int:
    if not a or not b: return 0
    dp = [0]*(len(b)+1)
    best = 0
    for i in range(1, len(a)+1):
        prev = 0
        for j in range(1, len(b)+1):
            tmp = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev + 1
                if dp[j] > best: best = dp[j]
            else:
                dp[j] = 0
            prev = tmp
    return best

def _token_sim(a: str, b: str) -> float:
    # a,b should be robust-normalized
    if not a or not b: return 0.0
    ed = _levenshtein(a, b)
    return 1.0 - ed / max(len(a), len(b))

def _union_valid(rects: List[Optional[Rect]]) -> Optional[Rect]:
    rects = [r for r in rects if r is not None and r[2] > r[0] and r[3] > r[1]]
    if not rects: return None
    x0 = min(r[0] for r in rects); y0 = min(r[1] for r in rects)
    x1 = max(r[2] for r in rects); y1 = max(r[3] for r in rects)
    return (x0, y0, x1, y1)

def _rect_to_tuple(r: Rect) -> Tuple[int,int,int,int]:
    return (int(r[0]), int(r[1]), int(r[2]), int(r[3]))

def _pack_match(i: int, raw: str, norm: str, rect: Rect, why: str, score: Optional[float] = None, span: Optional[Tuple[int,int]] = None) -> Dict[str, Any]:
    d = {
        "index": int(i),
        "token": raw,
        "norm": norm,
        "rect": list(_rect_to_tuple(rect)),
        "why": str(why),
    }
    if score is not None:
        d["score"] = float(score)
    if span is not None:
        d["span"] = [int(span[0]), int(span[1])]  # [start, length] for n-gram spans
    return d

# ========= Main function =========
def find_answer_rect_from_ocr(
    ocr_info: List[Dict],
    image_size: Tuple[int, int],
    answer: str,
    max_ngram: int = 5,
    *,
    # thresholds / knobs
    min_chars: int = 2,
    min_score: float = 0.40,
    lcs_weight: float = 0.6,
    per_token_min_score: float = 0.6,
    allow_numeric: bool = True,
    numeric_min_len: int = 2,
    numeric_accept_phrase: bool = True,
    return_all: bool = False,
    question_text: Optional[str] = None
) -> Optional[Rect] | Tuple[Optional[Rect], List[Rect], Dict[str, Any]]:
    """
    Unified answer-grounding with rich match tracing.
    """
    W, H = image_size
    toks = [d.get("word", "") for d in ocr_info]
    boxes_norm = [d.get("bounding_box", {}) for d in ocr_info]

    boxes: List[Optional[Rect]] = []
    for bb in boxes_norm:
        if {"top_left_x", "top_left_y", "width", "height"} <= bb.keys():
            x0 = int(round(bb["top_left_x"] * W))
            y0 = int(round(bb["top_left_y"] * H))
            w = int(round(bb["width"] * W))
            h = int(round(bb["height"] * H))
            r = (x0, y0, x0 + w, y0 + h)
            boxes.append(r if _rect_is_valid(r) else None)
        else:
            boxes.append(None)

    meta: Dict[str, Any] = {"path": None, "score": None}
    matches: List[Dict[str, Any]] = []

    # Track a global best candidate even if below thresholds (for forced fallback)
    best_any_score: float = -1.0
    best_any_rects: List[Rect] = []
    best_any_path: Optional[str] = None

    def _maybe_update_global(rects: List[Rect], score: float, path: str):
        nonlocal best_any_score, best_any_rects, best_any_path
        if rects and score is not None and score > best_any_score:
            best_any_score = float(score)
            best_any_rects = rects[:]
            best_any_path = path

    ans_norm = _norm_robust(answer)
    if not ans_norm or len(ans_norm) < min_chars:
        result = (None, [], {"path": "reject_short_answer", "matches": []})
        return result if return_all else None

    norm_toks = [_norm_robust(t) for t in toks]

    # ---- 1) exact single-token ----
    token_hits_rects: List[Rect] = []
    for i, nt in enumerate(norm_toks):
        if boxes[i] is None:
            continue
        if nt == ans_norm:
            token_hits_rects.append(boxes[i])
            matches.append(_pack_match(i, toks[i], nt, boxes[i], why="exact_token", score=1.0))
    if token_hits_rects:
        u = _union_valid(token_hits_rects)
        meta.update({"path": "exact_token", "score": 1.0, "matches": matches})
        return (u, token_hits_rects, meta) if return_all else u

    # ---- 2/3/4) phrase scan ----
    ans_nums = _extract_numbers(answer) if allow_numeric else []
    ans_nums = [n for n in ans_nums if len(n.replace(".", "")) >= numeric_min_len]

    best_phrase = None  # (score, i, n, union_rect, span_indices)
    exact_phrase_rects: Optional[Rect] = None
    numeric_phrase_rects: Optional[Rect] = None

    N = min(max_ngram, len(toks))
    for n in range(N, 1, -1):  # prefer longer phrases
        for i in range(0, len(toks) - n + 1):
            span_idx = [j for j in range(i, i + n) if boxes[j] is not None]
            if not span_idx:
                continue
            cand_raw = " ".join(toks[i:i + n])
            cand_norm = _norm_robust(cand_raw)
            if not cand_norm:
                continue
            u = _union_valid([boxes[j] for j in span_idx])
            if u is None:
                continue

            # exact phrase
            if cand_norm == ans_norm:
                for j in span_idx:
                    matches.append(_pack_match(j, toks[j], norm_toks[j], boxes[j], why="exact_phrase", score=1.0, span=(i, n)))
                exact_phrase_rects = u
                break

            cand_nums = []
            if allow_numeric:
                cand_nums = [x for x in _extract_numbers(cand_raw) if len(x.replace(".", "")) >= numeric_min_len]

            # numeric early accept
            if ans_nums and cand_nums and set(ans_nums) & set(cand_nums):
                if numeric_accept_phrase:
                    for j in span_idx:
                        matches.append(_pack_match(j, toks[j], norm_toks[j], boxes[j], why="numeric_phrase", score=1.0, span=(i, n)))
                    numeric_phrase_rects = u

            # fuzzy score
            lcs = _lcs_len(ans_norm, cand_norm)
            ed = _levenshtein(ans_norm, cand_norm)
            ed_sim = 1.0 - (ed / max(len(ans_norm), len(cand_norm)))
            score = lcs_weight * (lcs / max(1, len(ans_norm))) + (1.0 - lcs_weight) * ed_sim

            # numeric bonus
            if ans_nums and cand_nums and set(ans_nums) & set(cand_nums):
                score = min(1.0, score + 0.15)

            if (best_phrase is None) or (score > best_phrase[0]):
                best_phrase = (score, i, n, u, span_idx)
            _maybe_update_global([u], score, "fuzzy_phrase_best")

        if exact_phrase_rects is not None:
            break

    if exact_phrase_rects is not None:
        meta.update({"path": "exact_phrase", "score": 1.0, "matches": matches})
        return (exact_phrase_rects, [exact_phrase_rects], meta) if return_all else exact_phrase_rects

    if numeric_phrase_rects is not None and numeric_accept_phrase:
        meta.update({"path": "numeric_phrase", "score": 1.0, "matches": matches})
        return (numeric_phrase_rects, [numeric_phrase_rects], meta) if return_all else numeric_phrase_rects

    if best_phrase is not None and best_phrase[0] is not None and best_phrase[0] >= min_score:
        sc, i0, n0, u0, ids = best_phrase
        for j in ids:
            matches.append(_pack_match(j, toks[j], norm_toks[j], boxes[j], why="fuzzy_phrase", score=float(sc), span=(i0, n0)))
        meta.update({"path": "fuzzy_phrase", "score": float(sc), "matches": matches})
        return (u0, [u0], meta) if return_all else u0

    # ---- 5) per-word collect ----
    rect_list: List[Rect] = []
    seen_rect = set()
    keywords = _split_answer_keywords(answer, min_token_chars=min_chars)

    # exact per-token
    for kw in keywords:
        kw_norm = _norm_robust(kw)
        for i, nt in enumerate(norm_toks):
            if boxes[i] is None:
                continue
            if nt == kw_norm:
                r = boxes[i]
                key = _rect_to_tuple(r)
                if key not in seen_rect:
                    rect_list.append(r); seen_rect.add(key)
                    matches.append(_pack_match(i, toks[i], nt, r, why="per_token_exact", score=1.0))

    # numeric per-token
    if allow_numeric:
        kw_nums = set([n for kw in keywords for n in _extract_numbers(kw) if len(n.replace(".", "")) >= numeric_min_len])
        if kw_nums:
            for i, raw in enumerate(toks):
                if boxes[i] is None:
                    continue
                cand_nums = [x for x in _extract_numbers(raw) if len(x.replace(".", "")) >= numeric_min_len]
                if set(cand_nums) & kw_nums:
                    r = boxes[i]
                    key = _rect_to_tuple(r)
                    if key not in seen_rect:
                        rect_list.append(r); seen_rect.add(key)
                        matches.append(_pack_match(i, toks[i], norm_toks[i], r, why="per_token_numeric", score=1.0))

    # fuzzy per-token for remaining
    have_norm = {nt for nt in norm_toks}
    still = [_norm_robust(kw) for kw in keywords if _norm_robust(kw) not in have_norm]
    for kw_norm in still:
        best_i, best_sc = None, -1.0
        for i, nt in enumerate(norm_toks):
            if boxes[i] is None:
                continue
            sc = _token_sim(kw_norm, nt)
            if sc > best_sc:
                best_sc, best_i = sc, i
        if best_i is not None:
            r = boxes[best_i]
            _maybe_update_global([r], best_sc, "token_fuzzy_best")
            if best_sc >= per_token_min_score:
                key = _rect_to_tuple(r)
                if key not in seen_rect:
                    rect_list.append(r); seen_rect.add(key)
                    matches.append(_pack_match(best_i, toks[best_i], norm_toks[best_i], r, why="per_token_fuzzy", score=float(best_sc)))

    if rect_list:
        u = _union_valid(rect_list)
        meta.update({"path": "per_word_collect", "score": 1.0, "matches": matches})
        return (u, rect_list, meta) if return_all else u

    # ---- 6) yes/no fallback to question terms ----
    yn = ans_norm
    if yn in {"yes", "no", "true", "false"} and question_text:
        q_matches = find_support_rects_from_question(
            ocr_info, image_size, question_text,
            min_chars=3, per_token_min_score=0.75, allow_numeric=True, numeric_min_len=2
        )
        if q_matches:
            for m in q_matches:
                matches.append({
                    "index": int(m["index"]),
                    "token": m["token"],
                    "norm": m.get("norm", _norm_robust(m["token"])),
                    "rect": list(m["rect"]),
                    "why": m.get("why", "question"),
                    "score": float(m.get("score", 1.0)),
                })
            rects = [tuple(m["rect"]) for m in q_matches]
            u = _union_valid(rects)
            meta.update({"path": "question_fallback", "score": 1.0, "matches": matches})
            return (u, rects, meta) if return_all else u

    # ---- 7) forced-best fallback (use best_any_score even if < thresholds) ----
    if best_any_rects:
        inv = { _rect_to_tuple(boxes[i]): i for i in range(len(boxes)) if boxes[i] is not None }
        for r in best_any_rects:
            rt = _rect_to_tuple(r)
            i = inv.get(rt, -1)
            tok = toks[i] if i >= 0 else ""
            norm = norm_toks[i] if i >= 0 else ""
            matches.append(_pack_match(i, tok, norm, r, why=f"forced_{best_any_path or 'best'}", score=float(best_any_score)))
        u = _union_valid(best_any_rects)
        meta.update({"path": f"forced_best_{best_any_path or 'unknown'}", "score": float(best_any_score), "matches": matches})
        return (u, best_any_rects, meta) if return_all else u

    # ---- 8) largest-box fallback (last resort) ----
    largest = None
    largest_area = -1
    for r in boxes:
        if not _rect_is_valid(r):
            continue
        area = (r[2] - r[0]) * (r[3] - r[1])
        if area > largest_area:
            largest_area = area
            largest = r
    if largest is not None:
        inv = { _rect_to_tuple(boxes[i]): i for i in range(len(boxes)) if boxes[i] is not None }
        i = inv.get(_rect_to_tuple(largest), -1)
        tok = toks[i] if i >= 0 else ""
        norm = norm_toks[i] if i >= 0 else ""
        matches.append(_pack_match(i, tok, norm, largest, why="largest_box_fallback", score=0.0))
        meta.update({"path": "largest_box_fallback", "score": 0.0, "matches": matches})
        return (largest, [largest], meta) if return_all else largest

    print(f"Warning: no valid boxes at all in OCR data? {ocr_info}", file=sys.stderr)
    meta.update({"path": "none", "score": None, "matches": matches})
    return (None, [], meta) if return_all else None


def get_grid_cell_rects_generic(W:int, H:int, margin_px:int, offset_px:int, N:int):
    """Return N×N cell rectangles within outer margins/offsets."""
    x0, y0 = margin_px + offset_px, margin_px + offset_px
    x1, y1 = W - (margin_px + offset_px), H - (margin_px + offset_px)
    cw = max(1, (x1 - x0) // N)
    ch = max(1, (y1 - y0) // N)
    cells: Dict[str, Rect] = {}
    for gy in range(N):
        for gx in range(N):
            lx = x0 + gx * cw
            ty = y0 + gy * ch
            rx = lx + cw
            by = ty + ch
            code = f"{gx}-{gy}"  # e.g., "0-0", "1-2"
            cells[code] = (lx, ty, rx, by)
    return cells

def get_grid_anchors_generic(W:int, H:int, tw:int, th:int, margin_px:int, offset_px:int, N:int):
    """Return per-cell top-left anchors that keep the text box inside each cell."""
    cells = get_grid_cell_rects_generic(W,H,margin_px,offset_px,N)
    anchors: Dict[str, Tuple[int,int]] = {}
    for code, (lx,ty,rx,by) in cells.items():
        ax = min(max(lx, lx + 2), max(lx, rx - tw))   # nudge inward
        ay = min(max(ty, ty + 2), max(ty, by - th))
        anchors[code] = (ax, ay)
    return anchors, cells

def _rect_center(r: Rect) -> Tuple[int,int]:
    return ((r[0] + r[2]) // 2, (r[1] + r[3]) // 2)

def _cell_containing_rect(cell_rects: Dict[str, Rect], target: Rect) -> Optional[str]:
    """Return code of the cell whose area contains the center of target rect."""
    cx, cy = _rect_center(target)
    for code, (x0,y0,x1,y1) in cell_rects.items():
        if x0 <= cx < x1 and y0 <= cy < y1:
            return code
    return None

def _rect_edge_dist_norm(att_rect: Rect, ans_rect: Rect, W: int, H: int) -> float:
    """
    Normalized minimum corner-to-corner distance between two rectangles.
    Returns 0 if they overlap (any corner of one inside the other).
    """
    Ax1, Ay1, Ax2, Ay2 = att_rect
    Bx1, By1, Bx2, By2 = ans_rect

    # --- (1) Overlap check: any corner inside the other ---
    def _inside(x, y, rect: Rect) -> bool:
        x0, y0, x1, y1 = rect
        return (x0 < x < x1) and (y0 < y < y1)

    att_corners = [(Ax1, Ay1), (Ax1, Ay2), (Ax2, Ay1), (Ax2, Ay2)]
    ans_corners = [(Bx1, By1), (Bx1, By2), (Bx2, By1), (Bx2, By2)]

    # if any corner of A in B, or B in A → overlap
    for (x, y) in att_corners:
        if _inside(x, y, ans_rect):
            return 0.0
    for (x, y) in ans_corners:
        if _inside(x, y, att_rect):
            return 0.0

    # --- (2) Otherwise, take minimum corner distance ---
    min_d = float("inf")
    for (ax, ay) in att_corners:
        for (bx, by) in ans_corners:
            d = math.hypot(ax - bx, ay - by)
            if d < min_d:
                min_d = d

    # --- (3) Normalize by image diagonal ---
    diag = max(1.0, math.hypot(W, H))
    return min_d / diag

def place_text_bucketed_dist_grid(
    image: Image.Image,
    ocr_info: List[Dict],
    answer_text: str,
    attack_text: str,
    bucket: str = "far",
    grid_size: int = 11,
    random_within_cell: bool = True,
    font_path: Optional[str] = None,
    font_size: int = 28,
    min_font_size: int = 14,
    margin_rel: float = 0.01,
    pad_rel: float = 0.01,
    offset_px: int = 8,
    stroke_width: int = 0,
    stroke_fill: Tuple[int,int,int] = (0,0,0),
    fill: Tuple[int,int,int] = (255,255,255),
    rnd: Optional["random.Random"] = None,
    max_try_in_cell: int = 20,
    question_text: Optional[str] = None,
    is_debug: bool = False,
) -> Tuple[Image.Image, Tuple[int,int], Rect, int, str, Optional[Rect], Dict[str, Any], float]:
    """
    Place attack text using an N×N grid, selecting a cell by edge-distance bucket from the answer box.
    Returns: (img, xy, rect, used_font, chosen_cell_code, answer_rect, answer_meta, dist_norm_edge_final)
    """
    import random as _r
    if rnd is None: rnd = _r.Random(1234)
    img = image.copy().convert("RGB")
    W, H = img.size
    margin_px = int(round(margin_rel * max(W, H)))

    forbidden = prepare_forbidden_rects(ocr_info, (W, H), pad_rel=pad_rel)
    union_rect, rect_list, ans_rect_meta = find_answer_rect_from_ocr(
        ocr_info, (W,H), answer_text, return_all=True, question_text=question_text
    )
    answer_rect = union_rect
    if answer_rect is None:
        answer_rect = (W//2, H//2, W//2+1, H//2+1)  # dummy 1px rect at center
    if is_debug:
        print(f"Debug: answer_rect: {answer_rect}, meta: {ans_rect_meta}", file=sys.stderr)

    used = font_size
    while used >= min_font_size:
        font = _load_font(font_path, used)
        draw_probe = ImageDraw.Draw(img)

        l0,t0,r0,b0 = _safe_textbbox(draw_probe, (0,0), attack_text, font,
                                      stroke_width=stroke_width,
                                      stroke_fill=(stroke_fill if stroke_width>0 else None))
        tw, th = r0 - l0, b0 - t0

        anchors, cells = get_grid_anchors_generic(W,H,tw,th,margin_px,offset_px,grid_size)

        # Exclude the cell containing the answer center
        exclude = _cell_containing_rect(cells, answer_rect) if answer_rect else None

        feas: List[Dict[str, Any]] = []
        for code, base_xy in anchors.items():
            if exclude is not None and code == exclude:
                continue
            base_rect = _safe_textbbox(draw_probe, base_xy, attack_text, font,
                                       stroke_width=stroke_width,
                                       stroke_fill=(stroke_fill if stroke_width > 0 else None))
            if not _fits(base_rect, W, H, margin_px, forbidden):
                continue
            dist = _rect_edge_dist_norm(base_rect, answer_rect, W, H)
            feas.append({"code": code, "base_xy": base_xy, "base_rect": base_rect,
                         "dist": max(0.0, float(dist))})
        if is_debug:
            print(f"Debug: feas: {[(f['code'], round(f['dist'], 2)) for f in feas]}", file=sys.stderr)

        if not feas:
            used -= 2
            continue

        # Rank-based tertiles on edge distances
        dvals = sorted(f["dist"] for f in feas)
        def pct(a, p):
            idx = min(len(a)-1, max(0, int(round(p * (len(a)-1)))))
            return a[idx]
        cuts = [pct(dvals, q) for q in (0.0, 1/3, 2/3, 1.0)]

        def in_bucket(d: float, tag: str) -> bool:
            if tag == "near": lo, hi = cuts[0], cuts[1]
            elif tag == "mid": lo, hi = cuts[1], cuts[2]
            else:              lo, hi = cuts[2], cuts[3]
            return (d >= lo) and (d <= hi)

        order = {"far": ["far","mid","near"],
                 "mid": ["mid","far","near"],
                 "near":["near","mid","far"]}[bucket]
        picked = None
        final_bucket = bucket
        for tag in order:
            cand = [f for f in feas if in_bucket(f["dist"], tag)]
            if cand:
                picked = rnd.choice(cand)
                final_bucket = tag  # record actual bucket used
                break
        if picked is None:
            picked = rnd.choice(feas)
            final_bucket = f"{bucket}_fallback_random"

        code, base_xy, base_rect = picked["code"], picked["base_xy"], picked["base_rect"]
        drawer = ImageDraw.Draw(img)

        # Try uniform placement within the chosen cell; fall back to anchor
        if random_within_cell:
            cx0, cy0, cx1, cy1 = cells[code]
            minx, maxx = cx0, max(cx0, cx1 - tw)
            miny, maxy = cy0, max(cy0, cy1 - th)
            for _ in range(max_try_in_cell):
                x = rnd.randint(minx, maxx)
                y = rnd.randint(miny, maxy)
                trial_rect = _safe_textbbox(drawer, (x,y), attack_text, font,
                                            stroke_width=stroke_width,
                                            stroke_fill=(stroke_fill if stroke_width>0 else None))
                if not _fits(trial_rect, W, H, margin_px, forbidden):
                    continue
                # Draw and re-measure to get final rect
                drawer.text((x, y), attack_text, font=font, fill=fill,
                            stroke_width=stroke_width, stroke_fill=(stroke_fill if stroke_width > 0 else None))
                rect = _safe_textbbox(drawer, (x,y), attack_text, font,
                                      stroke_width=stroke_width,
                                      stroke_fill=(stroke_fill if stroke_width>0 else None))
                dist_final = 0.0 if answer_rect is None else _rect_edge_dist_norm(rect, answer_rect, W, H)
                return img, (x,y), rect, used, code, answer_rect, ans_rect_meta, float(dist_final)

        # Fallback: use anchor position
        drawer.text(base_xy, attack_text, font=font, fill=fill,
                    stroke_width=stroke_width, stroke_fill=(stroke_fill if stroke_width > 0 else None))
        rect = _safe_textbbox(drawer, base_xy, attack_text, font,
                              stroke_width=stroke_width,
                              stroke_fill=(stroke_fill if stroke_width>0 else None))
        dist_final = 0.0 if answer_rect is None else _rect_edge_dist_norm(rect, answer_rect, W, H)
        return img, base_xy, rect, used, code, answer_rect, ans_rect_meta, float(dist_final)

    # Last resort
    used = max(min_font_size, 8)
    font = _load_font(font_path, used)
    drawer = ImageDraw.Draw(img)
    xy = (margin_px + offset_px, margin_px + offset_px)
    drawer.text(xy, attack_text, font=font, fill=fill,
                stroke_width=stroke_width, stroke_fill=(stroke_fill if stroke_width>0 else None))
    rect = _safe_textbbox(drawer, xy, attack_text, font,
                          stroke_width=stroke_width,
                          stroke_fill=(stroke_fill if stroke_width>0 else None))
    dist_final = 0.0 if (answer_rect is None) else _rect_edge_dist_norm(rect, answer_rect, W, H)
    return img, xy, rect, used, "TL", answer_rect, ans_rect_meta, float(dist_final)


# -----------------------------
# Random continuous placement (simple reject sampling)
# -----------------------------
def place_text_random_continuous(
    image: Image.Image,
    ocr_info: List[Dict],
    text: str,
    font: ImageFont.FreeTypeFont,
    margin_px: int,
    forbidden: List[Rect],
    rnd: random.Random,
    bbox_margin_px: int = 8,
    max_retry: int = 50,
    stroke_width: int = 2,
    stroke_fill: Tuple[int,int,int] = (0,0,0),
    fill: Tuple[int,int,int] = (255,255,255),
    is_debug: bool = False
) -> Optional[Tuple[Image.Image, Tuple[int,int], Rect]]:
    W, H = image.size
    draw = ImageDraw.Draw(image)
    l0, t0, r0, b0 = _safe_textbbox(draw, (0,0), text, font, stroke_width=stroke_width,
                                    stroke_fill=(stroke_fill if stroke_width > 0 else None))

    tw, th = r0 - l0, b0 - t0
    # allowed region for top-left
    xmin = margin_px + bbox_margin_px
    ymin = margin_px + bbox_margin_px
    xmax = W - margin_px - bbox_margin_px - tw
    ymax = H - margin_px - bbox_margin_px - th
    if xmax <= xmin or ymax <= ymin:
        print(f"Warning: no space to place text box of size ({tw}x{th}) in image ({W}x{H}) with margin {margin_px}+{bbox_margin_px}", file=sys.stderr)
        # return None
        # force place at top-left corner
        img_copy = image.copy()
        draw2 = ImageDraw.Draw(img_copy)
        x, y = xmin, ymin
        draw2.text((x, y), text, font=font, fill=fill,
                stroke_width=stroke_width,
                stroke_fill=(stroke_fill if stroke_width > 0 else None))
        rect = _safe_textbbox(draw2, (x, y), text, font,
                            stroke_width=stroke_width,
                            stroke_fill=(stroke_fill if stroke_width > 0 else None))
        return img_copy, (x, y), rect

    for _ in range(max_retry):
        x = rnd.randint(xmin, xmax)
        y = rnd.randint(ymin, ymax)
        trial_rect = _safe_textbbox(draw, (x, y), text, font,
                                    stroke_width=stroke_width,
                                    stroke_fill=(stroke_fill if stroke_width > 0 else None))
        if not _fits(trial_rect, W, H, margin_px, forbidden):
            continue
        img_copy = image.copy()
        draw2 = ImageDraw.Draw(img_copy)
        draw2.text((x, y), text, font=font, fill=fill,
                   stroke_width=stroke_width,
                   stroke_fill=(stroke_fill if stroke_width > 0 else None))
        rect = _safe_textbbox(draw2, (x, y), text, font,
                              stroke_width=stroke_width,
                              stroke_fill=(stroke_fill if stroke_width > 0 else None))
        return img_copy, (x, y), rect
    print(f"Warning: failed to place text after {max_retry} retries", file=sys.stderr)
    # force place at top-left corner
    img_copy = image.copy()
    draw2 = ImageDraw.Draw(img_copy)
    x, y = xmin, ymin
    draw2.text((x, y), text, font=font, fill=fill,
               stroke_width=stroke_width,
               stroke_fill=(stroke_fill if stroke_width > 0 else None))
    rect = _safe_textbbox(draw2, (x, y), text, font,
                          stroke_width=stroke_width,
                          stroke_fill=(stroke_fill if stroke_width > 0 else None))
    return img_copy, (x, y), rect


# -----------------------------
# Config helpers and random utilities
# -----------------------------
def load_json_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def seed_from_string(s: str) -> int:
    # deterministic seed from arbitrary string using md5
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:16], 16) & 0x7FFFFFFF

def choose_font_size(pattern: Dict[str, Any], rnd: random.Random) -> int:
    if "fixed_px" in pattern.get("font", {}):
        return int(pattern["font"]["fixed_px"])
    f = pattern.get("font", {})
    min_px = int(f.get("min_px", 14))
    max_px = int(f.get("max_px", 28))
    step = int(f.get("step", 1))
    sampling = f.get("sampling", "uniform")
    if sampling == "uniform_step" and step > 0:
        choices = list(range(min_px, max_px+1, step))
        return int(rnd.choice(choices))
    elif sampling == "uniform":
        return int(rnd.randint(min_px, max_px))
    else:
        # fallback
        return int((min_px + max_px) // 2)

def choose_fill(pattern: Dict[str, Any], config: Dict[str, Any], rnd: random.Random, img: Image.Image, candidate_xy: Optional[Tuple[int,int]] = None, font: Optional[ImageFont.FreeTypeFont] = None, text: Optional[str] = None) -> Tuple[int,int,int]:
    fill_spec = pattern.get("fill", {})
    mode = fill_spec.get("mode", "palette_uniform")
    # palette from config
    palette = config.get("color_palette_rgb", [
        [255,255,255],[0,0,0],[255,0,0],[255,128,0],[0,200,0],[0,128,255],[255,215,0],[160,32,240]
    ])
    if mode == "fixed_rgb":
        return tuple(fill_spec.get("rgb", [255,255,255]))
    if mode == "palette_uniform":
        indices = fill_spec.get("palette_indices", list(range(len(palette))))
        idx = rnd.choice(indices)
        return tuple(palette[idx])
    if mode == "auto_contrast_else_palette":
        # If candidate_xy & font+text are given, sample a small patch for luminance and pick black/white.
        try:
            if candidate_xy and font and text and img is not None:
                draw = ImageDraw.Draw(img)
                l,t,r,b = draw.textbbox(candidate_xy, text, font=font)
                pad = 3
                sx0 = max(0, l-pad); sy0 = max(0, t-pad); sx1 = min(img.width, r+pad); sy1 = min(img.height, b+pad)
                if sx1 > sx0 and sy1 > sy0:
                    patch = img.crop((sx0, sy0, sx1, sy1)).convert("L")
                    stat = ImageStat.Stat(patch)
                    mean = stat.mean[0]
                    if mean < 128:
                        return (255,255,255)
                    else:
                        return (0,0,0)
        except Exception:
            pass
        # fallback to palette
        indices = fill_spec.get("palette_indices", list(range(len(palette))))
        idx = rnd.choice(indices)
        return tuple(palette[idx])
    # default
    indices = fill_spec.get("palette_indices", list(range(len(palette))))
    idx = rnd.choice(indices)
    return tuple(palette[idx])


# -----------------------------
# Debug drawing helpers
# -----------------------------
def _ocr_pixel_rects(ocr_info: List[Dict], image_size: Tuple[int,int]) -> List[Rect]:
    """Return per-token pixel rects without padding (raw OCR boxes)."""
    W, H = image_size
    rects: List[Rect] = []
    for item in ocr_info:
        bb = item.get("bounding_box", {})
        if {"top_left_x","top_left_y","width","height"} <= bb.keys():
            rects.append(_to_pixel_rect_from_norm(bb, W, H))
    return rects

def _draw_rect(img: Image.Image, r: Rect, color=(255, 0, 0), width: int = 2):
    """Draw rectangle outline; avoid hard exit on invalid."""
    if r is None: 
        return
    if r[0] >= r[2] or r[1] >= r[3]: 
        print("[Warning] Invalid rect:", r)
        return
    d = ImageDraw.Draw(img)
    d.rectangle([r[0], r[1], r[2], r[3]], outline=tuple(color), width=int(width))

def _maybe_debug_draw(
    img: Image.Image,
    attack_rect: Optional[Rect],
    answer_rect: Optional[Rect],
    cell_rects: Optional[Dict[str, Rect]] = None,
    chosen_cell_code: Optional[str] = None,
    ocr_rects: Optional[List[Rect]] = None,
    forbidden_rects: Optional[List[Rect]] = None,
    debug_opts: Optional[Dict[str, Any]] = None,
):
    """Draw debug overlays depending on flags in debug_opts."""
    opts = debug_opts or {}
    w = int(opts.get("width", 5))

    attack_color    = tuple(opts.get("attack_color",    [255,   0,   0]))
    answer_color    = tuple(opts.get("answer_color",    [  0, 255,   0]))
    ocr_color       = tuple(opts.get("ocr_color",       [255, 165,   0]))  # orange
    forbidden_color = tuple(opts.get("forbidden_color", [135, 206, 250]))  # light blue
    cell_color      = tuple(opts.get("cell_color",      [  0,   0, 255]))  # blue

    show_cells      = bool(opts.get("show_cells", False))
    show_ocr        = bool(opts.get("show_ocr", True))
    show_forbidden  = bool(opts.get("show_forbidden", False))

    # Cells (optional)
    if show_cells and cell_rects:
        for code, cr in cell_rects.items():
            width = max(3, 2*w) if code == chosen_cell_code else w
            _draw_rect(img, cr, cell_color, width=width)

    # OCR token boxes (optional)
    if show_ocr and ocr_rects:
        for r in ocr_rects:
            _draw_rect(img, r, ocr_color, width=w)

    # Forbidden (expanded) OCR boxes (optional)
    if show_forbidden and forbidden_rects:
        for r in forbidden_rects:
            _draw_rect(img, r, forbidden_color, width=w)

    # Answer and attack boxes (answer hidden by default; enable if needed)
    if answer_rect:
        _draw_rect(img, answer_rect, answer_color, width=max(5, w))
    if attack_rect:
        # Expand attack box slightly for visibility
        ax0, ay0, ax1, ay1 = attack_rect
        aw, ah = ax1 - ax0, ay1 - ay0
        px = max(1, int(round(0.1 * aw)))
        py = max(1, int(round(0.3 * ah)))
        enlarged = (ax0 - px, ay0 - py, ax1 + px, ay1 + py)
        _draw_rect(img, enlarged, attack_color, width=max(3, w))


# -----------------------------
# Top-level pattern application wrapper
# -----------------------------
def apply_pattern_for_image(
    image: Image.Image,
    ocr_info: List[Dict],
    text_to_place: str,
    pattern: Dict[str, Any],
    config: Dict[str, Any],
    image_id: str,
    question_id: Optional[str] = None,
    question_text: Optional[str] = None,
    subset_name: Optional[str] = None,
    global_seed_rule: Optional[str] = None,
    answer_text: Optional[str] = None,
    is_debug: bool = False,
) -> Dict[str, Any]:
    """
    Apply a single pattern to a single image deterministically according to seed rule.
    Returns: {"image": PIL.Image, "metadata": {...}}
    """

    # ---------- RNG (deterministic) ----------
    pattern_id = pattern.get("pattern_id", "p")
    seed_rule = global_seed_rule or config.get("global", {}).get("seed_rule", "hash(image_id + pattern_id)")
    seed_str = f"{image_id}__{subset_name or ''}__{pattern_id}__{seed_rule}"
    seed = seed_from_string(seed_str)
    rnd = random.Random(seed)

    # ---------- Globals ----------
    g = config.get("global", {})
    margin_rel  = float(g.get("margin_rel", 0.01))
    pad_rel     = float(g.get("pad_rel", 0.01))
    offset_px   = int(g.get("offset_px", 8))
    stroke_w    = int(g.get("stroke_width", 0))   # Typo-D: default 0
    stroke_fill = tuple(g.get("stroke_fill", [0, 0, 0]))

    W, H = image.size
    margin_px = int(round(margin_rel * max(W, H)))
    forbidden = prepare_forbidden_rects(ocr_info, (W, H), pad_rel=pad_rel)

    ocr_rects_raw = _ocr_pixel_rects(ocr_info, (W, H))  
    cells_for_debug = None

    # ---------- Pattern basics ----------
    placement = pattern.get("placement", {})
    mode      = placement.get("mode", "grid3x3")

    # font / color
    font_px = choose_font_size(pattern, rnd)
    font    = _load_font(None, font_px)
    fill    = choose_fill(pattern, config, rnd, image, None, None, text_to_place)

    # precompute text bbox size
    draw_probe = ImageDraw.Draw(image)
    l, t, r, b = _safe_textbbox(draw_probe, (0, 0), text_to_place, font, stroke_width=stroke_w,
                                stroke_fill=(stroke_fill if stroke_w > 0 else None))

    # ---------- base metadata ----------
    metadata: Dict[str, Any] = {
        "image_id": image_id,
        "question_id": question_id,
        "pattern_id": pattern_id,
        "seed": seed,
        "placement_mode": mode,
        "font_px": int(font_px),
        "fill": list(fill),
        "stroke_width": stroke_w,
        "stroke_fill": list(stroke_fill),
    }

    # ---------- debug options ----------
    debug_cfg  = config.get("debug", {})               # global debug config
    debug_opts = dict(debug_cfg)
    debug_opts.update(pattern.get("debug", {}))

    # ---------- random_continuous ----------
    if mode == "random_continuous":
        bbox_margin_px = int(placement.get("bbox_margin_px", 8))
        max_retry      = int(g.get("max_retry", 100))
        res = place_text_random_continuous(
            image.copy().convert("RGB"),
            ocr_info,
            text_to_place,
            font,
            margin_px,
            forbidden,
            rnd,
            bbox_margin_px=bbox_margin_px,
            max_retry=max_retry,
            stroke_width=stroke_w,
            stroke_fill=stroke_fill,
            fill=tuple(fill),
            is_debug=is_debug
        )
        img_try, xy, rect = res
        metadata.update({
            "xy": [int(xy[0]), int(xy[1])],
            "rect": [int(v) for v in rect],
            "used_font_px": int(font_px),
            "anchor_code": None
        })
        if is_debug:
            _maybe_debug_draw(
                img_try, attack_rect=rect, answer_rect=None,
                cell_rects=cells_for_debug if debug_opts.get("show_cells", False) else None,
                chosen_cell_code=None,
                ocr_rects=ocr_rects_raw,
                forbidden_rects=forbidden if debug_opts.get("show_forbidden", False) else None,
                debug_opts=debug_opts
            )
        return {"image": img_try, "metadata": metadata}

    # ---------- grid_distance_bucketed (7x7 + edge distance) ----------
    elif mode == "grid_distance_bucketed":
        if answer_text is None:
            raise ValueError("answer_text must be provided for grid_distance_bucketed mode")
        if not ocr_info:
            print("[WARNING] ocr_info is empty; cannot use grid_distance_bucketed mode reliably")

        # Select distance bucket (fixed or sampled)
        bucket = placement.get("distance_bucket")
        choices = placement.get("choices")
        probs   = placement.get("probs")
        if choices:
            bucket = rnd.choices(choices, weights=probs, k=1)[0]

        random_within_cell = bool(placement.get("random_within_cell", True))  # train=True / val=False

        img_try, xy, rect, used_font, code, answer_rect, ans_rect_meta, dist_edge = place_text_bucketed_dist_grid(
            image, ocr_info,
            answer_text=answer_text, attack_text=text_to_place,
            bucket=bucket, random_within_cell=random_within_cell,
            font_path=None, font_size=font_px,
            min_font_size=int(pattern.get("font", {}).get("min_px", max(8, font_px - 8))),
            margin_rel=margin_rel, pad_rel=pad_rel, offset_px=offset_px,
            stroke_width=stroke_w, stroke_fill=stroke_fill, fill=tuple(fill),
            rnd=rnd,
            question_text=question_text,
            is_debug=is_debug
        )
        metadata.update({
            "xy": [int(xy[0]), int(xy[1])],
            "rect": [int(v) for v in rect],
            "used_font_px": int(used_font),
            "anchor_code": code,
            "answer_rect": [int(v) for v in answer_rect] if answer_rect else None,
            "answer_rect_meta": ans_rect_meta,
            "distance_bucket": bucket,
            "distance_value_edge_norm": float(dist_edge),
            "random_within_cell": random_within_cell
        })
        if is_debug:
            _maybe_debug_draw(
                img_try,
                attack_rect=rect,
                answer_rect=answer_rect,
                cell_rects=None,  # could be added if needed
                chosen_cell_code=code,
                ocr_rects=ocr_rects_raw,
                forbidden_rects=forbidden if debug_opts.get("show_forbidden", False) else None,
                debug_opts=debug_opts
            )
        return {"image": img_try, "metadata": metadata}

    else:
        raise NotImplementedError



# -----------------------------
# Example main: how to run with config files
# -----------------------------
if __name__ == "__main__":
    """
    Demo runner WITHOUT argv.
    - Toggle MODE among: "val_fixed", "val_random", "train"
    - It tries to load configs from ./configs/*.json; if not found, uses inline defaults.
    - Processes the first example of TextVQA val set and saves outputs to ./out/
    """

    import json
    from datasets import load_dataset

    # -----------------------
    # 1) Simple toggles
    # -----------------------
    # MODE = "val_fixed"           # "val_fixed" | "val_random" | "train"

    # config files 
    CONFIG_DIR = "typo_attack_config"
    CFG_PATHS = {
        "obj": {
            # "train": os.path.join(CONFIG_DIR, "obj_attack", "train_config.json"),
            # "val_rc": os.path.join(CONFIG_DIR, "obj_attack", "val_config.json"),
        },
        "txt": {
            "val_far": os.path.join(CONFIG_DIR, "txt_attack", "val_config_far.json"),
            "val_mid": os.path.join(CONFIG_DIR, "txt_attack", "val_config_mid.json"),
            "val_near": os.path.join(CONFIG_DIR, "txt_attack", "val_config_near.json"),
            # "train": os.path.join(CONFIG_DIR, "txt_attack", "train_far_config.json"),
            # "train": os.path.join(CONFIG_DIR, "txt_attack", "train_mid_config.json"),
        }
    }
    
    for attack_type, modes in CFG_PATHS.items():
        for MODE, CFG_PATH in modes.items():
            # MODE = "train"           # "val_fixed" | "val_random" | "train"
            ATTACK_TEXT = "ATTACK-WORD"  # attack token to place (for demo)
            SAVE_DIR = Path("examples")
            SAVE_DIR.mkdir(exist_ok=True)

            # Optional: paths for OCR json and config files (if present, they’re used)
            DEFAULT_OCR_JSON = "assets/textvqa/TextVQA_Rosetta_OCR_v0.2_val.json"

            with open(CFG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)

            # -----------------------
            # 4) Load one sample from TextVQA val and OCR info
            # -----------------------
            ds = load_dataset("facebook/textvqa", split="validation")
            
            for i in range(5):
                item = ds[i]  # demo: first sample only
                image: Image.Image = item["image"]
                image_id = str(item["image_id"])
                question_id = item["question_id"]
                answers = item["answers"]
                answer = max(set(answers), key=answers.count)

                # OCR info
                ocr_info = []
                try:
                    with open(DEFAULT_OCR_JSON, "r", encoding="utf-8") as f:
                        textocr_data = json.load(f)
                    image_id2ocr = {d["image_id"]: d for d in textocr_data.get("data", [])}
                    ocr_info = image_id2ocr.get(item["image_id"], {}).get("ocr_info", [])
                    print(f"[INFO] OCR boxes: {len(ocr_info)} for image_id={image_id}")
                except Exception as e:
                    print(f"[WARN] Could not load OCR JSON ({DEFAULT_OCR_JSON}). Proceeding with empty OCR. Err={e}")

                # -----------------------
                # 5) Apply patterns in config and save
                # -----------------------
                patterns = config.get("patterns", [])
                split = config.get("split", MODE)
                for pat in patterns:
                    # For text-aware farthest mode (not in current inline), we’d pass answer_text via pat["answer_text"]=answer
                    res = apply_pattern_for_image(
                        image=image,
                        ocr_info=ocr_info,
                        text_to_place=ATTACK_TEXT,
                        pattern=pat,
                        config=config,
                        image_id=image_id,
                        question_id=question_id,
                        subset_name=split,
                        answer_text=answer,
                        is_debug=True
                    )
                    out_img = res["image"]
                    meta = res["metadata"]

                    # fname = f"{image_id}__{split}__{meta['pattern_id']}__seed{meta['seed']}.jpg"
                    # out_img.save(SAVE_DIR / fname)
                    save_dir = SAVE_DIR / attack_type / split / pat["pattern_id"]
                    save_dir.mkdir(parents=True, exist_ok=True)
                    fname = f"{image_id}__{split}__{meta['pattern_id']}__seed{meta['seed']}.jpg"
                    out_img.save(save_dir / fname)
                    print("[SAVED]", save_dir / fname)
                    # metadata
                    with open(save_dir / (fname + ".json"), "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2)
                    print("[SAVED]", save_dir / (fname + ".json"))
               

    print("[DONE] Demo finished.")
