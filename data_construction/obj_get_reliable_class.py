import os
import json
import pickle
from typing import Dict, List, Set, Tuple, Iterable, Optional
from functools import lru_cache

import torch
import numpy as np
from PIL import Image
import PIL
import open_clip
from datasets import load_dataset

# ---------- (1) 祖先を除去して leaf 候補だけ残す ----------
def _all_ancs(label: str, abs_ancestors: Dict[str, Dict[int, Set[str]]]) -> Set[str]:
    out = set()
    for _, S in abs_ancestors.get(label, {}).items():
        out |= S
    return out

def prune_to_leaves(candidates: List[str],
                    abs_ancestors: Dict[str, Dict[int, Set[str]]]) -> List[str]:
    keep = []
    cand_set = set(candidates)
    for a in candidates:
        is_ancestor = any(a in _all_ancs(b, abs_ancestors) for b in cand_set if b != a)
        if not is_ancestor:
            keep.append(a)
    return keep or candidates

# ---------- (2) CLIP バッチ推論 ----------
class ClipScorer:
    def __init__(
        self,
        model_id: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        templates: Tuple[str, ...] = (
            "a photo of a {}.",
            "an image of a {}.",
            "a close-up photo of a {}.",
            "a cropped photo of a {}.",
            "a low angle photo of a {}.",
            "a high angle photo of a {}.",
        ),
        device: Optional[str] = None,
        text_batch_size: int = 256,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_id, pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_id)
        self.templates = templates
        self.text_batch_size = text_batch_size

    @torch.no_grad()
    def encode_images(self, images: List[PIL.Image.Image], image_batch_size: int = 64) -> torch.Tensor:
        """Return normalized image features [N, D]."""
        feats = []
        for i in range(0, len(images), image_batch_size):
            batch = images[i:i+image_batch_size]
            ims = torch.stack([self.preprocess(im) for im in batch]).to(self.device)
            f = self.model.encode_image(ims)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f)
        return torch.cat(feats, dim=0)

    def _encode_texts_in_chunks(self, texts: List[str]) -> torch.Tensor:
        outs = []
        for i in range(0, len(texts), self.text_batch_size):
            tok = self.tokenizer(texts[i:i+self.text_batch_size]).to(self.device)
            t = self.model.encode_text(tok)
            t = t / t.norm(dim=-1, keepdim=True)
            outs.append(t)
        return torch.cat(outs, dim=0)

    @lru_cache(maxsize=4096)
    @torch.no_grad()
    def encode_labels_avg_templates(self, labels_key: Tuple[str, ...]) -> torch.Tensor:
        """
        テンプレート平均済みのラベル埋め込みを返す: [L, D]
        labels_key はタプル（キャッシュ用）。順序保存・重複除去しない前提で OK。
        """
        labels = list(labels_key)
        # すべての (template x label) を一括で作成し、エンコード → 平均
        all_prompts = []
        for tmpl in self.templates:
            all_prompts.extend([tmpl.format(l) for l in labels])
        T = len(self.templates)
        L = len(labels)
        txt = self._encode_texts_in_chunks(all_prompts)  # [T*L, D]
        txt = txt.view(T, L, -1).mean(dim=0)             # [L, D] テンプレート平均
        txt = txt / txt.norm(dim=-1, keepdim=True)
        return txt  # [L, D]

    @torch.no_grad()
    def score_batch(
        self,
        images: List[PIL.Image.Image],
        labels_list: List[List[str]],
        image_batch_size: int = 64,
    ) -> Tuple[List[str], List[Dict[str, float]]]:
        """
        images: バッチ画像
        labels_list: 各サンプルごとの候補ラベル（順序保持）
        戻り値: best_labels, all_scores (各サンプルの {label: score})
        """
        assert len(images) == len(labels_list)
        img_feats = self.encode_images(images, image_batch_size=image_batch_size)  # [N, D]

        best_labels: List[str] = []
        all_scores: List[Dict[str, float]] = []

        for i, labels in enumerate(labels_list):
            if len(labels) == 0:
                best_labels.append("")
                all_scores.append({})
                continue

            # キャッシュキー（順序付きタプル化）
            key = tuple(labels)
            txt_feats = self.encode_labels_avg_templates(key)  # [L, D]

            sims = (img_feats[i:i+1] @ txt_feats.T).squeeze(0).detach().float().cpu().numpy()  # [L]
            scores_dict = {l: float(s) for l, s in zip(labels, sims)}
            # 降順並べ替え
            scores_sorted = dict(sorted(scores_dict.items(), key=lambda x: x[1], reverse=True))
            best_labels.append(labels[int(np.argmax(sims))])
            all_scores.append(scores_sorted)

        return best_labels, all_scores

# ---------- (3) データセット → バッチ推論ドライバ ----------
def batched(iterable: Iterable, n: int):
    """Yield lists of length <= n."""
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

if __name__ == "__main__":
    ASSETS_DIR = "assets/open_images"

    with open(os.path.join(ASSETS_DIR, "abs_ancestors.pkl"), "rb") as f:
        abs_ancestors = pickle.load(f)
    with open(os.path.join(ASSETS_DIR, "all_ancestors.pkl"), "rb") as f:
        all_ancestors = pickle.load(f)
    with open(os.path.join(ASSETS_DIR, "parent2children.pkl"), "rb") as f:
        parent2children = pickle.load(f)
    print("loaded abs_ancestors, all_ancestors, parent2children")

    split = "train"  # or "validation"
    print(f"Generating reliable class labels for TextVQA {split} set")
    ds = load_dataset("facebook/textvqa", split=split)

    scorer = ClipScorer(
        model_id="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        templates=(
            "a photo of a {}.",
            "an image of a {}.",
            "a close-up photo of a {}.",
            "a cropped photo of a {}.",
            "a low angle photo of a {}.",
            "a high angle photo of a {}.",
        ),
        device=None,
        text_batch_size=256,
    )

    out_path = f"textvqa_{split}_question_id2best_label.json"
    question_id2best_label = {}

    # 途中再実行対応：既存結果があれば読み込む
    if os.path.exists(out_path):
        try:
            with open(out_path, "r") as f:
                question_id2best_label = json.load(f)
            print(f"resume from existing {out_path}: {len(question_id2best_label)} entries")
        except Exception:
            pass

    BATCH = 64            # 画像のエンコードバッチ
    SAVE_EVERY = 1000     # 何サンプルごとに保存するか

    total = len(ds)
    processed = 0 if not question_id2best_label else len(question_id2best_label)

    # 既に処理済みの question_id はスキップできるようにする
    done_qids = set(question_id2best_label.keys())

    buffer_images: List[PIL.Image.Image] = []
    buffer_labels: List[List[str]] = []
    buffer_qids: List[str] = []
    save_counter = 0

    for item in ds:
        qid = item["question_id"]
        if qid in done_qids:
            continue

        image: PIL.Image.Image = item["image"]
        candidates: List[str] = item["image_classes"]
        leaves = prune_to_leaves(candidates, abs_ancestors)

        buffer_images.append(image)
        buffer_labels.append(leaves)
        buffer_qids.append(qid)

        # 画像側のミニバッチが貯まったら推論
        if len(buffer_images) >= BATCH:
            bests, scores = scorer.score_batch(buffer_images, buffer_labels, image_batch_size=BATCH)
            for q, b, s in zip(buffer_qids, bests, scores):
                question_id2best_label[q] = {"best_label": b, "all_scores": s}
            processed += len(buffer_qids)
            save_counter += len(buffer_qids)

            if save_counter >= SAVE_EVERY:
                with open(out_path, "w") as f:
                    json.dump(question_id2best_label, f, indent=2)
                print(f"saved {out_path} ({processed}/{total})")
                save_counter = 0

            buffer_images.clear()
            buffer_labels.clear()
            buffer_qids.clear()

    # 端数を処理
    if buffer_images:
        bests, scores = scorer.score_batch(buffer_images, buffer_labels, image_batch_size=BATCH)
        for q, b, s in zip(buffer_qids, bests, scores):
            question_id2best_label[q] = {"best_label": b, "all_scores": s}
        processed += len(buffer_qids)

    # 最終保存
    with open(out_path, "w") as f:
        json.dump(question_id2best_label, f, indent=2)
    print(f"done. saved {out_path} ({processed}/{total})")