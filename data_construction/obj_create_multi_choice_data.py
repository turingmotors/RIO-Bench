from typing import Dict, Set, List, Tuple, Optional, Any
import random
from collections import deque
from tqdm import tqdm


# --- Select y_true -----------------------------------------------------------
def node_depth(label: str, abs_ancestors: Dict[str, Dict[int, Set[str]]]) -> int:
    """Depth(node) = 1 + max ancestor depth; if unknown, return 1."""
    depth_map = abs_ancestors.get(label, {})
    return 1 + max(depth_map.keys()) if depth_map else 1


def is_ancestor(a: str, b: str, abs_ancestors: Dict[str, Dict[int, Set[str]]]) -> bool:
    """
    Return True if `a` is an ancestor of `b` under the absolute-depth ancestry map.

    abs_ancestors[b]: depth -> set of ancestor labels at that depth
    """
    for ancestors_at_depth in abs_ancestors.get(b, {}).values():
        if a in ancestors_at_depth:
            return True
    return False


def prune_gt_to_pseudo_leaves(
    gt_labels: Set[str],
    abs_ancestors: Dict[str, Dict[int, Set[str]]],
) -> Set[str]:
    """
    Within the GT set, drop any label that is an ancestor of another GT label,
    i.e., keep only the "deepest" members inside the GT set.
    """
    keep: Set[str] = set(gt_labels)
    for x in list(gt_labels):
        for y in list(gt_labels):
            if x == y:
                continue
            if is_ancestor(x, y, abs_ancestors):  # x is ancestor of y
                # Drop the ancestor x, keep the child y
                if x in keep:
                    keep.remove(x)
    return keep


def pick_best_label(
    candidates: Set[str],
    all_scores: Dict[str, float],
) -> Optional[Tuple[str, float]]:
    """
    Pick the highest-scoring label from `candidates` using `all_scores`.

    If none of the candidates has a score, return None.
    Tie-break deterministically by label name (ascending).
    """
    scored = [(lbl, all_scores[lbl]) for lbl in candidates if lbl in all_scores]
    if not scored:
        return None
    # sort by (-score, label) then take first
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[0]


def select_y_true_for_one(
    all_scores: Dict[str, float],
    gt_labels: Set[str],
    abs_ancestors: Dict[str, Dict[int, Set[str]]],
    min_depth_preference: int = 3,
    target_root: Optional[str] = "Entity",
) -> Dict[str, Any]:
    """
    Select y_true for a single image/question.

    Policy:
      1) If parent and child both appear in GT, drop the parent (GT pruning to pseudo-leaves).
      2) Among the remaining GT labels, if any label has depth >= min_depth_preference
         (default 3), restrict to those; otherwise use all remaining GT labels.
      3) Choose the label with the maximum CLIP score from all_scores.

      Fallback:
        - If no GT label has a score, pick the argmax from all_scores
          (optionally restricted to labels under target_root if available).
        - If all_scores is empty, return y_true=None, score=None.

    Returns:
        {
          "y_true": str or None,
          "score": float or None,
          "reason": one of
            "picked_depth>=3" |
            "picked_depth<=2" |
            "fallback_gt_empty" |
            "fallback_any" |
            "no_scores"
        }
    """

    # Optional: require candidates to be under the designated root at depth=1
    def under_root(lbl: str) -> bool:
        if target_root is None:
            return True
        return target_root in abs_ancestors.get(lbl, {}).get(1, set())

    # Step 0: filter GT by root (conservative)
    gt_rooted = {g for g in gt_labels if under_root(g)} if gt_labels else set()

    # Step 1: prune GT to pseudo-leaves (drop ancestors that have their children in GT)
    gt_pruned = prune_gt_to_pseudo_leaves(gt_rooted, abs_ancestors) if gt_rooted else set()

    # If pruning removes everything (or GT is empty), fall back to rooted GT;
    # if that is also empty, we will fall back later.
    gt_for_scoring = gt_pruned if gt_pruned else gt_rooted

    # Partition by depth (>= min_depth_preference vs shallower)
    deep = {
        g for g in gt_for_scoring
        if node_depth(g, abs_ancestors) >= min_depth_preference
    }
    shallow = gt_for_scoring - deep

    # Step 2 & 3: pick by CLIP score from deep first, then shallow
    if deep:
        pick = pick_best_label(deep, all_scores)
        if pick is not None:
            lbl, sc = pick
            return {"y_true": lbl, "score": sc, "reason": "picked_depth>=3"}

    if shallow:
        pick = pick_best_label(shallow, all_scores)
        if pick is not None:
            lbl, sc = pick
            return {"y_true": lbl, "score": sc, "reason": "picked_depth<=2"}

    # Fallbacks: no GT label present in scores → pick best from all_scores
    if not all_scores:
        return {"y_true": None, "score": None, "reason": "no_scores"}

    # Prefer labels under root if any exist in all_scores; otherwise use all labels
    rooted_all = {l: s for l, s in all_scores.items() if under_root(l)}
    pool = rooted_all if rooted_all else all_scores
    lbl, sc = sorted(pool.items(), key=lambda x: (-x[1], x[0]))[0]
    reason = "fallback_gt_empty" if gt_labels else "fallback_any"
    return {"y_true": lbl, "score": sc, "reason": reason}


def select_y_true_batch(
    qid2_scores: Dict[str, Dict[str, float]],
    qid2_gt: Dict[str, List[str]],
    abs_ancestors: Dict[str, Dict[int, Set[str]]],
    min_depth_preference: int = 3,
    target_root: Optional[str] = "Entity",
) -> Dict[str, Dict[str, Any]]:
    """
    Batch wrapper for y_true selection.

    Args:
        qid2_scores: qid -> {label: score, ...}
        qid2_gt:     qid -> [gt_label, ...]

    Returns:
        qid -> result of select_y_true_for_one(...)
    """
    out: Dict[str, Dict[str, Any]] = {}
    for qid, scores in tqdm(qid2_scores.items(), desc="Selecting y_true for batch"):
        gts = set(qid2_gt.get(qid, []))
        out[qid] = select_y_true_for_one(
            all_scores=scores,
            gt_labels=gts,
            abs_ancestors=abs_ancestors,
            min_depth_preference=min_depth_preference,
            target_root=target_root,
        )
    return out


# --- Sample negatives (hard/medium/easy) ------------------------------------
def sample_negatives_hme(
    y_true: str,
    gt_labels: Set[str],
    label_vocab: List[str],
    abs_ancestors: Dict[str, Dict[int, Set[str]]],   # depth:1 = root, increasing downward
    parent2children: Dict[str, Set[str]],            # direct edges parent -> children
    target_root: str = "Entity",
    seed: int = 42,
    seed_salt: Optional[str] = None,                 # e.g., image_id or question_id
) -> Dict[str, str]:
    """
    Sample hard/medium/easy negatives purely based on the hierarchy.

    Steps:
      - Assert (conceptually) that y_true is a leaf.
      - Prune GT to leaves (under the hierarchy).
      - Exclude:
          (pruned GT leaves) and ALL of their descendants.
        (Do NOT exclude GT ancestors.)

    Band definitions (same as the original design):
      hard   = siblings at the deepest parent level of y_true (share ancestors at depth L)
      medium = share ancestors at depth L-1 but NOT at depth L
      easy   = share ancestors at depth L-2 but NOT at depth L-1 or L

    The function always returns distinct labels for {"hard", "medium", "easy"}
    via staged back-offs and a final distinctness guard.

    Returns:
        {"hard": label, "medium": label, "easy": label}
    """

    # ---------- RNG seeding ----------
    if seed_salt is None:
        random.seed(seed)
    else:
        random.seed(f"{seed}-{seed_salt}")

    # ---------- Small helpers ----------
    def anc_map(lbl: str) -> Dict[int, Set[str]]:
        return abs_ancestors.get(lbl, {})

    def ancestors_at(lbl: str, depth: int) -> Set[str]:
        return set(anc_map(lbl).get(depth, set()))

    def all_anc(lbl: str) -> Set[str]:
        out = set()
        for _, s in anc_map(lbl).items():
            out |= s
        return out

    def has_root(lbl: str) -> bool:
        return target_root in ancestors_at(lbl, 1)

    def children_of(lbl: str) -> Set[str]:
        return set(parent2children.get(lbl, set()))

    def is_leaf(lbl: str) -> bool:
        return len(children_of(lbl)) == 0

    def descendants_of(lbl: str) -> Set[str]:
        """Collect all descendants via BFS over parent2children."""
        out: Set[str] = set()
        q = deque(children_of(lbl))
        while q:
            u = q.popleft()
            if u in out:
                continue
            out.add(u)
            for v in children_of(u):
                q.append(v)
        return out

    # ---------- (0) conceptually require y_true to be a leaf ----------

    # ---------- (1) prune GT to leaves ----------
    pruned_gt = {g for g in gt_labels if is_leaf(g)}
    # ensure y_true is included (it should be a leaf conceptually)
    if y_true in gt_labels:
        pruned_gt.add(y_true)

    # ---------- (2) build exclusion set: pruned GT leaves and their descendants ----------
    exclude_set: Set[str] = set(pruned_gt)
    for g in pruned_gt:
        # leaves typically add none; non-leaf GT would add its subtree
        exclude_set |= descendants_of(g)

    # ---------- (3) compute deepest parent level L for y_true ----------
    depths = sorted([d for d in anc_map(y_true).keys() if d > 1])
    # If no depth > 1 exists (edge case), treat as shallow taxonomy
    if not depths:
        # Keep the original fallback that sets depths = [2] to
        # keep later formulas for (L, Lm, Le) consistent.
        depths = [2]
    L = depths[-1]
    Lm = max(2, L - 1)
    Le = max(2, L - 2)

    # ---------- (4) base candidate pool
    # Apply exclusion; do NOT remove ancestors of non-y_true GT labels ----------
    base_pool = [
        c
        for c in label_vocab
        if c != y_true
        and c not in exclude_set            # exclude pruned GT leaves and their descendants only
        and has_root(c)                     # stay under target root
    ]

    # ---------- (5) band ancestors for y_true ----------
    y_par_L = ancestors_at(y_true, L)
    y_par_Lm = ancestors_at(y_true, Lm)
    y_par_Le = ancestors_at(y_true, Le)

    # ---------- (6) band-specific pools ----------
    def pool_hard() -> List[str]:
        """Hard negatives: share any ancestor at depth L with y_true."""
        return [c for c in base_pool if ancestors_at(c, L) & y_par_L]

    def pool_medium() -> List[str]:
        """
        Medium negatives:
            share L-1 with y_true but NOT share L.
        """
        return [
            c
            for c in base_pool
            if (ancestors_at(c, Lm) & y_par_Lm)
            and not (ancestors_at(c, L) & y_par_L)
        ]

    def pool_easy() -> List[str]:
        """
        Easy negatives:
            share L-2 with y_true but NOT share L-1 or L.
        """
        return [
            c
            for c in base_pool
            if (ancestors_at(c, Le) & y_par_Le)
            and not (ancestors_at(c, Lm) & y_par_Lm)
            and not (ancestors_at(c, L) & y_par_L)
        ]

    hard_pool = pool_hard()
    medium_pool = pool_medium()
    easy_pool = pool_easy()

    # ---------- (7) pickers ----------
    def pick_from(pool: List[str], used: Set[str]) -> Optional[str]:
        """
        Pick a label from `pool` that is not in `used`.
        Pick uniformly at random.
        """
        candidates = [c for c in pool if c not in used]
        if not candidates:
            return None
        return random.choice(candidates)

    def pick_easy_uniform(used: Set[str]) -> Optional[str]:
        """
        Pick easy negative uniformly at random from easy_pool, or
        from base_pool if easy_pool is empty.
        """
        pool = [c for c in (easy_pool if easy_pool else base_pool) if c not in used]
        return random.choice(pool) if pool else None

    # ---------- (8) selection with staged back-offs ----------
    chosen: Dict[str, Optional[str]] = {"hard": None, "medium": None, "easy": None}
    used: Set[str] = set()

    # Hard: prefer hard pool, then medium, then global base_pool
    h = pick_from(hard_pool, used) or pick_from(medium_pool, used) or pick_from(base_pool, used)
    if h:
        chosen["hard"] = h
        used.add(h)

    # Medium: prefer medium pool, then easy, then global base_pool
    m = pick_from(medium_pool, used) or pick_from(easy_pool, used) or pick_from(base_pool, used)
    if m:
        chosen["medium"] = m
        used.add(m)

    # Easy: prefer easy (or base_pool if easy empty)
    e = pick_easy_uniform(used) or pick_from(base_pool, used)
    if e:
        chosen["easy"] = e
        used.add(e)

    # ---------- (9) final distinctness guard ----------
    vals = [chosen["hard"], chosen["medium"], chosen["easy"]]
    if len(set(vals)) < 3:
        # Try to replace duplicates using any remaining label from the global vocab
        pool = [
            c
            for c in label_vocab
            if c not in exclude_set and c != y_true and c not in set(vals)
        ]
        random.shuffle(pool)
        seen: Set[Optional[str]] = set()
        for key in ("hard", "medium", "easy"):
            if chosen[key] in seen and pool:
                chosen[key] = pool.pop()
            seen.add(chosen[key])

    # ---------- (10) corner-case fallback: ensure no None ----------
    for k in ("hard", "medium", "easy"):
        if chosen[k] is None:
            # pick anything not excluded / not used / not y_true
            fallback = [
                c
                for c in label_vocab
                if c not in exclude_set and c != y_true and c not in used
            ]
            chosen[k] = random.choice(fallback) if fallback else y_true  # last resort

    return {
        "hard": chosen["hard"],
        "medium": chosen["medium"],
        "easy": chosen["easy"],
    }


if __name__ == "__main__":
    import os
    import pickle
    import json
    from datasets import load_dataset

    ASSETS_DIR = "assets"
    OPEN_IMAGES_DIR = f"{ASSETS_DIR}/open_images"
    TEXTVQA_DIR = f"{ASSETS_DIR}/textvqa_meta"

    MCQ_DATA_DIR = f"{ASSETS_DIR}/mcq_data"
    os.makedirs(MCQ_DATA_DIR, exist_ok=True)

    # Load Open Images hierarchy pickles
    with open(f"{OPEN_IMAGES_DIR}/abs_ancestors.pkl", "rb") as f:
        abs_ancestors = pickle.load(f)
    with open(f"{OPEN_IMAGES_DIR}/all_ancestors.pkl", "rb") as f:
        all_ancestors = pickle.load(f)
    with open(f"{OPEN_IMAGES_DIR}/parent2children.pkl", "rb") as f:
        parent2children = pickle.load(f)
    print("Loaded abs_ancestors, all_ancestors, parent2children")

    # Build a mapping from label -> top-level branches (labels that have no ancestors)
    label2topbranch: Dict[str, List[str]] = {}
    for label, ancestors in all_ancestors.items():
        top_branches = [a for a in ancestors if a not in all_ancestors]
        label2topbranch[label] = top_branches
    print(f"Total {len(label2topbranch)} labels with top branches.")
    TOP = label2topbranch  # Kept for compatibility / debugging if needed

    # Vocab is defined as labels observed in the train split
    textvqa_vocab_path = f"{TEXTVQA_DIR}/textvqa_vocab.json"
    if os.path.exists(textvqa_vocab_path):
        with open(textvqa_vocab_path, "r") as f:
            vocab = json.load(f)
        print(f"Loaded TextVQA vocab from {textvqa_vocab_path}, size: {len(vocab)}")
    else:
        ds = load_dataset("facebook/textvqa", split="train")
        gt_labels_set: Set[str] = set()
        for item in ds:
            img_classes = item.get("image_classes") or []
            for c in img_classes:
                gt_labels_set.add(c)
        print(f"Collected {len(gt_labels_set)} unique GT labels from TextVQA train split.")

        vocab = list(gt_labels_set)
        with open(textvqa_vocab_path, "w") as f:
            json.dump(vocab, f)
        print(f"Saved TextVQA vocab to {textvqa_vocab_path}, size: {len(vocab)}")
    print(f"Vocab size: {len(vocab)}")

    # -------------------------------------------------------------------------
    # Small standalone example for a single y_true
    # -------------------------------------------------------------------------
    y_true = "Cat"
    gt_labels = {"Cat", "Dog", "Animal"}
    negs = sample_negatives_hme(
        y_true=y_true,
        gt_labels=gt_labels,
        label_vocab=vocab,
        abs_ancestors=abs_ancestors,
        parent2children=parent2children,
        seed=42,
    )
    print("Example negatives for", y_true, "with GT", gt_labels, "->", negs)

    # -------------------------------------------------------------------------
    # Full TextVQA dataset example
    # -------------------------------------------------------------------------
    for SPLIT in ["train", "validation"]:
        SPLIT_SHORT = "val" if SPLIT == "validation" else "train"

        out_path = os.path.join(MCQ_DATA_DIR, f"textvqa_{SPLIT_SHORT}_mcq_samples.json")
        if os.path.exists(out_path):
            print(f"MCQ samples for TextVQA {SPLIT} already exist at {out_path}, skipping.")
            continue

        ds = load_dataset("facebook/textvqa", split=SPLIT)
        print(f"Loaded TextVQA {SPLIT} split, num samples:", len(ds))

        # Use textvqa_{split}_question_id2best_label.json to determine y_true
        with open(f"{TEXTVQA_DIR}/textvqa_{SPLIT}_question_id2best_label.json", "r") as f:
            qid2info = json.load(f)
        print(
            f"Loaded {len(qid2info)} entries from "
            f"textvqa_{SPLIT}_question_id2best_label.json"
        )

        qid2y_true = select_y_true_batch(
            qid2_scores={qid: info["all_scores"] for qid, info in qid2info.items()},
            qid2_gt={item["question_id"]: item["image_classes"] for item in ds},
            abs_ancestors=abs_ancestors,
            min_depth_preference=3,
            target_root="Entity",
        )
        print(f"Selected y_true for {len(qid2y_true)} questions.")

        mc_samples_list: List[Dict[str, Any]] = []
        for i, sample in tqdm(enumerate(ds), desc="Generating MCQ samples"):
            image_classes = sample.get("image_classes") or []
            question_id = sample.get("question_id")

            # NOTE: This uses the original condition on `gt_labels` (left as-is for
            # behavioral compatibility; gt_labels here is the example set above).
            if not gt_labels:
                print(f"Warning: no GT labels for image {i}, skipping.")
                continue

            y_true_info = qid2y_true.get(str(question_id))
            y_true, reason = (
                (y_true_info.get("y_true"), y_true_info.get("reason"))
                if y_true_info
                else (None, "no_entry")
            )
            if not y_true or y_true not in vocab:
                print(
                    f"Warning: invalid y_true for image {i}, skipping. "
                    f"question_id={question_id}, y_true={y_true}, reason={reason}"
                )
                continue

            negs = sample_negatives_hme(
                y_true=y_true,
                gt_labels=gt_labels,
                label_vocab=vocab,
                abs_ancestors=abs_ancestors,
                parent2children=parent2children,
                seed=42,
                seed_salt=sample.get("question_id"),  # vary negatives per question
            )

            mc_samples = {
                "image_id": sample["image_id"],
                "question_id": sample["question_id"],
                "answer": y_true,
                "choices": {
                    "hard": negs["hard"],
                    "medium": negs["medium"],
                    "easy": negs["easy"],
                },
                "image_classes": image_classes,
            }
            mc_samples_list.append(mc_samples)

        print(f"Generated {len(mc_samples_list)} MCQ samples.")
        with open(out_path, "w") as f:
            json.dump(mc_samples_list, f, indent=2)
        print(f"Saved MCQ samples to {out_path}")

        print("Skipped sample length:", len(ds) - len(mc_samples_list))
