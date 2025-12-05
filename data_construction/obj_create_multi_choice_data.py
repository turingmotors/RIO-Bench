from typing import Dict, Set, List, Tuple, Optional, Any
import random
from collections import deque

# --- select y true ----
def node_depth(label: str, abs_ancestors: Dict[str, Dict[int, Set[str]]]) -> int:
    """Depth(node) = 1 + max ancestor depth; if unknown, return 1."""
    depth_map = abs_ancestors.get(label, {})
    return 1 + max(depth_map.keys()) if depth_map else 1

def is_ancestor(a: str, b: str, abs_ancestors: Dict[str, Dict[int, Set[str]]]) -> bool:
    """Return True if `a` is an ancestor of `b` under the absolute-depth ancestry map."""
    for dmap in abs_ancestors.get(b, {}).values():
        if a in dmap:
            return True
    return False

def prune_gt_to_pseudo_leaves(gt_labels: Set[str],
                              abs_ancestors: Dict[str, Dict[int, Set[str]]]) -> Set[str]:
    """
    Within the GT set, drop any label that is an ancestor of another GT label
    (keep only the deepest GT members).
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

def pick_best_label(candidates: Set[str],
                    all_scores: Dict[str, float]) -> Optional[Tuple[str, float]]:
    """
    Pick the highest-scoring label from `candidates` using `all_scores`.
    If none of the candidates has a score, return None.
    Tie-break deterministically by label name.
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
    Select y_true for a single image/question based on the policy:
      1) If parent and child both appear in GT, drop the parent (keep only children).
      2) If any remaining GT label has depth >= min_depth_preference (default 3),
         restrict to those; else use all remaining GT labels.
      3) Choose the label with the maximum CLIP score from all_scores.
      Fallback: if no GT label has a score, pick the max from all_scores
                (optionally restrict to labels under target_root if available).

    Returns:
      {
        "y_true": str or None,
        "score": float or None,
        "reason": "picked_depth>=3" | "picked_depth<=2" | "fallback_gt_empty" | "fallback_any" | "no_scores"
      }
    """
    # Optional: require candidates to be under the designated root at depth=1
    def under_root(lbl: str) -> bool:
        if target_root is None:
            return True
        return target_root in abs_ancestors.get(lbl, {}).get(1, set())

    # Step 0: root filtering for GT (conservative)
    gt_rooted = {g for g in gt_labels if under_root(g)} if gt_labels else set()

    # Step 1: prune GT to pseudo-leaves (drop ancestors that have their children in GT)
    gt_pruned = prune_gt_to_pseudo_leaves(gt_rooted, abs_ancestors) if gt_rooted else set()

    # If pruning removes everything (or GT empty), try raw GT; if still empty, we will fallback later
    gt_for_scoring = gt_pruned if gt_pruned else gt_rooted

    # Partition by depth (>=3 vs <=2)
    deep = {g for g in gt_for_scoring if node_depth(g, abs_ancestors) >= min_depth_preference}
    shallow = gt_for_scoring - deep

    # Step 2 & 3: pick by CLIP score
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

    # Fallbacks: no GT label present in scores → pick max from all_scores
    if not all_scores:
        return {"y_true": None, "score": None, "reason": "no_scores"}

    # Prefer labels under root if any exist in all_scores; otherwise use all
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
    Batch wrapper.
      qid2_scores: qid -> {"label": score, ...}
      qid2_gt:     qid -> [gt_label, ...]
    """
    out: Dict[str, Dict[str, Any]] = {}
    for qid, scores in qid2_scores.items():
        gts = set(qid2_gt.get(qid, []))
        out[qid] = select_y_true_for_one(
            all_scores=scores,
            gt_labels=gts,
            abs_ancestors=abs_ancestors,
            min_depth_preference=min_depth_preference,
            target_root=target_root,
        )
    return out

# --- sample negatives ---
def sample_negatives_hme(
    y_true: str,
    gt_labels: Set[str],
    label_vocab: List[str],
    abs_ancestors: Dict[str, Dict[int, Set[str]]],   # depth:1=root, increasing downward
    parent2children: Dict[str, Set[str]],            # direct edges parent -> children
    target_root: str = "Entity",
    counts: Optional[Dict[str, int]] = None,         # optional (used only for tie-breaks in fallbacks)
    seed: int = 42,
    seed_salt: Optional[str] = None,                 # e.g., image_id
) -> Dict[str, str]:
    """
    hard/medium/easy negatives purely by hierarchy, with GT pruning to leaves:
      - Assert y_true is a leaf
      - Prune GT to leaves
      - Exclude: (pruned GT leaves) and ALL their descendants
      - Do NOT exclude GT ancestors

    Band definitions (same as before):
      hard   = siblings at deepest parent level of y_true (share L)
      medium = share L-1 but NOT share L
      easy   = share L-2 but NOT share L-1/L

    Always returns distinct labels for {'hard','medium','easy'} via staged backoffs.
    """

    # ---------- RNG seeding ----------
    if seed_salt is None:
        random.seed(seed)
    else:
        random.seed(f"{seed}-{seed_salt}")

    # ---------- small helpers ----------
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
        """Collect all descendants via BFS."""
        out = set()
        q = deque(children_of(lbl))
        while q:
            u = q.popleft()
            if u in out: 
                continue
            out.add(u)
            for v in children_of(u):
                q.append(v)
        return out

    # ---------- (0) assert y_true is a leaf ----------
    # assert is_leaf(y_true), f"y_true must be a leaf, but '{y_true}' has children: {children_of(y_true)}"

    # ---------- (1) prune GT to leaves ----------
    pruned_gt = {g for g in gt_labels if is_leaf(g)}
    # ensure y_true is included (it should be a leaf by assert)
    if y_true in gt_labels:
        pruned_gt.add(y_true)

    # ---------- (2) build exclusion: pruned GT and their descendants ----------
    exclude_set = set(pruned_gt)
    for g in pruned_gt:
        exclude_set |= descendants_of(g)   # leaves typically add none; non-leaf GT would add its subtree

    # ---------- deepest parent level L for y_true ----------
    depths = sorted([d for d in anc_map(y_true).keys() if d > 1])
    # if no depth > 1 exists (edge case), treat as shallow taxonomy
    if not depths:
        depths = [2]  # so that L=2, Lm=max(2,1)=2, Le=max(2,0)=2 still work safely
    L  = depths[-1]
    Lm = max(2, L - 1)
    Le = max(2, L - 2)

    # ---------- base candidate pool (apply exclusion; do NOT remove ancestors of non-y_true GT) ----------
    base_pool = [
        c for c in label_vocab
        if c != y_true
        and c not in exclude_set            # exclude pruned GT leaves and their descendants only
        and has_root(c)                     # stay under target root
    ]

    # ---------- band ancestors ----------
    y_par_L  = ancestors_at(y_true, L)
    y_par_Lm = ancestors_at(y_true, Lm)
    y_par_Le = ancestors_at(y_true, Le)

    # ---------- pool builders ----------
    def pool_hard() -> List[str]:
        # Siblings at deepest parent level: share any ancestor at L
        return [c for c in base_pool if ancestors_at(c, L)  & y_par_L]

    def pool_medium() -> List[str]:
        # Share L-1, but NOT L
        return [c for c in base_pool
                if (ancestors_at(c, Lm) & y_par_Lm) and not (ancestors_at(c, L) & y_par_L)]

    def pool_easy() -> List[str]:
        # Share L-2, but NOT L-1 or L
        return [c for c in base_pool
                if (ancestors_at(c, Le) & y_par_Le)
                and not (ancestors_at(c, Lm) & y_par_Lm)
                and not (ancestors_at(c, L)  & y_par_L)]

    hard_pool   = pool_hard()
    medium_pool = pool_medium()
    easy_pool   = pool_easy()

    # ---------- pickers ----------
    def pick_from(pool: List[str], used: Set[str]) -> Optional[str]:
        cand = [c for c in pool if c not in used]
        if not cand:
            return None
        if counts:
            random.shuffle(cand)  # avoid deterministically picking same lowest-count
            cand.sort(key=lambda x: counts.get(x, 10**9))
            return cand[0]
        return random.choice(cand)

    def pick_easy_uniform(used: Set[str]) -> Optional[str]:
        P = [c for c in (easy_pool if easy_pool else base_pool) if c not in used]
        return random.choice(P) if P else None

    # ---------- selection with staged backoffs ----------
    chosen: Dict[str, Optional[str]] = {"hard": None, "medium": None, "easy": None}
    used: Set[str] = set()

    h = pick_from(hard_pool, used) or pick_from(medium_pool, used) or pick_from(base_pool, used)
    if h: chosen["hard"] = h; used.add(h)

    m = pick_from(medium_pool, used) or pick_from(easy_pool, used) or pick_from(base_pool, used)
    if m: chosen["medium"] = m; used.add(m)

    e = pick_easy_uniform(used) or pick_from(base_pool, used)
    if e: chosen["easy"] = e; used.add(e)

    # Final distinctness guard
    vals = [chosen["hard"], chosen["medium"], chosen["easy"]]
    if len(set(vals)) < 3:
        pool = [c for c in label_vocab if c not in exclude_set and c != y_true and c not in set(vals)]
        random.shuffle(pool)
        seen = set()
        for key in ("hard", "medium", "easy"):
            if chosen[key] in seen and pool:
                chosen[key] = pool.pop()
            seen.add(chosen[key])

    # Convert Optional[str] → str (extreme corner case fallback)
    for k in ("hard", "medium", "easy"):
        if chosen[k] is None:
            # pick anything not excluded / not used / not y_true
            fallback = [c for c in label_vocab if c not in exclude_set and c != y_true and c not in used]
            chosen[k] = random.choice(fallback) if fallback else y_true  # last resort

    return {"hard": chosen["hard"], "medium": chosen["medium"], "easy": chosen["easy"]}


if __name__ == "__main__":
    import pickle
    import json
    
    ASSETS_DIR = "assets/open_images"

    # Open Image のヒエラルキー JSON を読み込み
    with open(f"{ASSETS_DIR}/abs_ancestors.pkl", "rb") as f:
        abs_ancestors = pickle.load(f)
    with open(f"{ASSETS_DIR}/all_ancestors.pkl", "rb") as f:
        all_ancestors = pickle.load(f)
    with open(f"{ASSETS_DIR}/parent2children.pkl", "rb") as f:
        parent2children = pickle.load(f)
    print("loaded abs_ancestors, all_ancestors, parent2children")

    label2topbranch = {}
    for label, ancestors in all_ancestors.items():
        top_branches = [a for a in ancestors if a not in all_ancestors]
        label2topbranch[label] = top_branches
    print(f"Total {len(label2topbranch)} labels with top branches.")
    TOP = label2topbranch

    # 例: train のクラス頻度と各画像のGTラベル（共起は train のみから作成）
    with open("textvqa_train_class2count.json", "r") as f:
        counts = json.load(f)
    print(f"Loaded class2count for {len(counts)} classes.")

    with open("textvqa_train_image_gt_lists.json", "r") as f:
        image_gt_lists = json.load(f)
    print(f"Loaded {len(image_gt_lists)} images' GT label lists.")

    # vocab は train で観測されたラベルとする（または固定語彙を別途用意）
    vocab = list(counts.keys())
    print(f"Vocab size: {len(vocab)}")

    # example
    y_true = "Cat"
    gt_labels = {"Cat", "Dog", "Animal"}
    negs = sample_negatives_hme(
        y_true=y_true,
        gt_labels=gt_labels,
        label_vocab=vocab,
        abs_ancestors=abs_ancestors,
        parent2children=parent2children,
        # counts=counts,
        seed=42,
    )
    print("Example negatives for", y_true, "with GT", gt_labels, "->", negs)

    # -----------------------------
    # TextVQA の全体データセットで試す場合
    # -----------------------------
    from datasets import load_dataset

    SPLIT = "train"  # or "train"
    SPLIT_SHORT = "val" if SPLIT == "validation" else "train"

    ds = load_dataset("facebook/textvqa", split=SPLIT)
    print("Loaded TextVQA validation split, num samples:", len(ds))

    # textvqa_val_question_id2best_label.json を使って y_true を決定する
    with open(f"textvqa_{SPLIT_SHORT}_question_id2best_label.json", "r") as f:
        qid2info = json.load(f)
    print(f"Loaded {len(qid2info)} entries from textvqa_{SPLIT_SHORT}_question_id2best_label.json")

    qid2y_true = select_y_true_batch(
        qid2_scores={qid: info["all_scores"] for qid, info in qid2info.items()},
        qid2_gt={item["question_id"]: item["image_classes"] for item in ds},
        abs_ancestors=abs_ancestors,
        min_depth_preference=3,
        target_root="Entity",
    )
    print(f"Selected y_true for {len(qid2y_true)} questions.")

    # print(qid2y_true)
    # exit()

    mc_samples_list = []
    for i, sample in enumerate(ds):
        if i % 100 == 0:
            print(f"Processing sample {i}/{len(ds)}")
        image_classes = sample.get("image_classes") or []
        question_id = sample.get("question_id")

        if not gt_labels:
            print(f"Warning: no GT labels for image {i}, skipping.")
            continue
        y_true_info = qid2y_true.get(str(question_id))
        y_true, reason = y_true_info.get("y_true"), y_true_info.get("reason") if y_true_info else (None, "no_entry")
        if not y_true or y_true not in vocab:
            print(f"Warning: invalid y_true for image {i}, skipping. question_id={question_id}, y_true={y_true}, reason={reason}")
            continue
        # if reason != "picked_min_depth":
        #     print(f"Warning: y_true for image {i} not a picked_min_depth, skipping. question_id={question_id}, y_true={y_true}, reason={reason}")
        #     print("  image_classes:", image_classes)
        #     continue
        negs = sample_negatives_hme(
            y_true=y_true,
            gt_labels=gt_labels,
            label_vocab=vocab,
            abs_ancestors=abs_ancestors,
            parent2children=parent2children,
            # counts=counts,
            seed=42,
            seed_salt=sample.get("question_id"),  # to vary negatives per question
        )
        # print(y_true, "->", negs)
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
    out_path = f"textvqa_{SPLIT_SHORT}_mcq_samples.json"
    with open(out_path, "w") as f:
        json.dump(mc_samples_list, f, indent=2)
    print(f"Saved MCQ samples to {out_path}")

    print("Skipped sample length:", len(ds) - len(mc_samples_list))
