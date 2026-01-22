import os
import json
import pickle
from typing import Dict, List, Set, Tuple, Iterable, Optional
from functools import lru_cache
from tqdm import tqdm

import torch
import numpy as np
from PIL import Image
import PIL
import open_clip
from datasets import load_dataset


# ---------- (1) Ancestor pruning: keep only leaf-like candidates ----------
def _all_ancs(label: str, abs_ancestors: Dict[str, Dict[int, Set[str]]]) -> Set[str]:
    """
    Collect all ancestors of a label across all depths.
    """
    out: Set[str] = set()
    for _, s in abs_ancestors.get(label, {}).items():
        out |= s
    return out


def prune_to_leaves(
    candidates: List[str],
    abs_ancestors: Dict[str, Dict[int, Set[str]]],
) -> List[str]:
    """
    Remove labels that are ancestors of other labels in `candidates`.

    If all candidates are ancestors of some other candidate, fall back to the
    original list to avoid returning an empty list.
    """
    keep: List[str] = []
    cand_set = set(candidates)

    for a in candidates:
        # Check if `a` is an ancestor of any other candidate
        is_ancestor = any(
            a in _all_ancs(b, abs_ancestors) for b in cand_set if b != a
        )
        if not is_ancestor:
            keep.append(a)

    # If everything was pruned (rare corner case), return original candidates
    return keep or candidates


# ---------- (2) CLIP batch scoring ----------
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
    ) -> None:
        """
        Wrapper for CLIP image/text encoding and similarity scoring.

        - model_id / pretrained: passed to open_clip.create_model_and_transforms
        - templates: multiple prompt templates per label, averaged in embedding space
        - device: None -> 'cuda' if available else 'cpu'
        - text_batch_size: max number of prompts per text-encoding batch
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_id, pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_id)
        self.templates = templates
        self.text_batch_size = text_batch_size

    @torch.no_grad()
    def encode_images(
        self,
        images: List[PIL.Image.Image],
        image_batch_size: int = 64,
    ) -> torch.Tensor:
        """
        Encode a list of PIL images into normalized CLIP image features.

        Returns:
            Tensor of shape [N, D], where N = len(images).
        """
        feats: List[torch.Tensor] = []
        for i in range(0, len(images), image_batch_size):
            batch = images[i : i + image_batch_size]
            ims = torch.stack([self.preprocess(im) for im in batch]).to(self.device)
            f = self.model.encode_image(ims)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f)
        return torch.cat(feats, dim=0)

    def _encode_texts_in_chunks(self, texts: List[str]) -> torch.Tensor:
        """
        Encode arbitrary number of text prompts in chunks, returning
        normalized text features stacked along rows.
        """
        outs: List[torch.Tensor] = []
        for i in range(0, len(texts), self.text_batch_size):
            tok = self.tokenizer(texts[i : i + self.text_batch_size]).to(self.device)
            t = self.model.encode_text(tok)
            t = t / t.norm(dim=-1, keepdim=True)
            outs.append(t)
        return torch.cat(outs, dim=0)

    @lru_cache(maxsize=4096)
    @torch.no_grad()
    def encode_labels_avg_templates(self, labels_key: Tuple[str, ...]) -> torch.Tensor:
        """
        Return template-averaged label embeddings of shape [L, D].

        Args:
            labels_key:
                A tuple of label strings (used as the cache key).
                We assume the tuple order is meaningful and do not deduplicate.

        For each label, we create prompts by filling all templates, encode all
        prompts, and average features across the template dimension.
        """
        labels = list(labels_key)
        # Build all (template x label) prompts, then encode and average.
        all_prompts: List[str] = []
        for tmpl in self.templates:
            all_prompts.extend([tmpl.format(l) for l in labels])

        T = len(self.templates)
        L = len(labels)

        txt = self._encode_texts_in_chunks(all_prompts)  # [T * L, D]
        txt = txt.view(T, L, -1).mean(dim=0)             # [L, D], averaged over templates
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
        Score a batch of images against per-sample candidate labels.

        Args:
            images:
                List of PIL images.
            labels_list:
                List of candidate-label lists, one list per image.
                (Order is preserved.)

        Returns:
            best_labels:
                List of the top-1 label for each image.
            all_scores:
                List of dictionaries mapping label -> similarity score
                for each image, sorted in descending order of score.
        """
        assert len(images) == len(labels_list)
        img_feats = self.encode_images(
            images, image_batch_size=image_batch_size
        )  # [N, D]

        best_labels: List[str] = []
        all_scores: List[Dict[str, float]] = []

        for i, labels in enumerate(labels_list):
            if len(labels) == 0:
                best_labels.append("")
                all_scores.append({})
                continue

            # Cache key: labels as an ordered tuple
            key = tuple(labels)
            txt_feats = self.encode_labels_avg_templates(key)  # [L, D]

            sims = (
                img_feats[i : i + 1] @ txt_feats.T
            ).squeeze(0).detach().float().cpu().numpy()  # [L]
            scores_dict = {l: float(s) for l, s in zip(labels, sims)}

            # Sort scores in descending order
            scores_sorted = dict(
                sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
            )

            best_labels.append(labels[int(np.argmax(sims))])
            all_scores.append(scores_sorted)

        return best_labels, all_scores


# ---------- (3) Dataset → batch inference driver ----------
if __name__ == "__main__":
    ASSETS_DIR = "assets/"
    OPEN_IMAGES_DIR = os.path.join(ASSETS_DIR, "open_images")
    OUT_DIR = os.path.join(ASSETS_DIR, "textvqa_meta")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Ancestor / hierarchy metadata
    with open(os.path.join(OPEN_IMAGES_DIR, "abs_ancestors.pkl"), "rb") as f:
        abs_ancestors = pickle.load(f)
    with open(os.path.join(OPEN_IMAGES_DIR, "all_ancestors.pkl"), "rb") as f:
        all_ancestors = pickle.load(f)
    with open(os.path.join(OPEN_IMAGES_DIR, "parent2children.pkl"), "rb") as f:
        parent2children = pickle.load(f)
    print("Loaded abs_ancestors, all_ancestors, parent2children")

    for split in ["train", "validation"]:
        out_path = os.path.join(OUT_DIR, f"textvqa_{split}_question_id2best_label.json")
        if os.path.exists(out_path):
            print(f"Output for {split} split already exists at {out_path}, skipping...")
            continue

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

        question_id2best_label: Dict[str, Dict[str, object]] = {}

        # Resume support: if the output already exists, load and append to it
        if os.path.exists(out_path):
            try:
                with open(out_path, "r") as f:
                    question_id2best_label = json.load(f)
                print(
                    f"Resuming from existing {out_path}: "
                    f"{len(question_id2best_label)} entries"
                )
            except Exception:
                # If loading fails, silently start from scratch (original behavior)
                pass

        BATCH = 64         # image encoding batch size
        SAVE_EVERY = 1000  # save after processing this many new samples

        total = len(ds)
        processed = 0 if not question_id2best_label else len(question_id2best_label)

        # Skip already processed question_ids if resuming
        done_qids = set(question_id2best_label.keys())

        buffer_images: List[PIL.Image.Image] = []
        buffer_labels: List[List[str]] = []
        buffer_qids: List[str] = []
        save_counter = 0

        for item in tqdm(ds):
            qid = item["question_id"]
            if qid in done_qids:
                continue

            image: PIL.Image.Image = item["image"]
            candidates: List[str] = item["image_classes"]

            # Prune ancestor labels so that only leaf-like labels remain
            leaves = prune_to_leaves(candidates, abs_ancestors)

            buffer_images.append(image)
            buffer_labels.append(leaves)
            buffer_qids.append(qid)

            # Run inference whenever we have a full image batch
            if len(buffer_images) >= BATCH:
                bests, scores = scorer.score_batch(
                    buffer_images, buffer_labels, image_batch_size=BATCH
                )
                for q, b, s in zip(buffer_qids, bests, scores):
                    question_id2best_label[q] = {"best_label": b, "all_scores": s}
                processed += len(buffer_qids)
                save_counter += len(buffer_qids)

                # Periodic checkpoint saving
                if save_counter >= SAVE_EVERY:
                    with open(out_path, "w") as f:
                        json.dump(question_id2best_label, f, indent=2)
                    print(f"Saved {out_path} ({processed}/{total})")
                    save_counter = 0

                buffer_images.clear()
                buffer_labels.clear()
                buffer_qids.clear()

        # Process any remaining samples that did not fill up the last batch
        if buffer_images:
            bests, scores = scorer.score_batch(
                buffer_images, buffer_labels, image_batch_size=BATCH
            )
            for q, b, s in zip(buffer_qids, bests, scores):
                question_id2best_label[q] = {"best_label": b, "all_scores": s}
            processed += len(buffer_qids)

        # Final save
        with open(out_path, "w") as f:
            json.dump(question_id2best_label, f, indent=2)
        print(f"Done. Saved {out_path} ({processed}/{total})")
