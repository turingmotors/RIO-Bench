"""
Simplified verision of CLIP-Match:
https://github.com/lmb-freiburg/ovqa/blob/main/ovqa/metrics/clip_match.py
"""
from typing import List, Dict, Any, Tuple, Optional
import torch
import open_clip
import sys, os
sys.path.append(os.path.dirname(__file__))

from cls_templates import CLASSIFICATION_TEMPLATES
TEMPLATES = CLASSIFICATION_TEMPLATES["openai_imagenet_template"]


def _templ(c: str) -> List[str]:
    """Expand a single concept into multiple template prompts."""
    c = c.replace("_", " ")
    return [tmpl(c) for tmpl in TEMPLATES]

@torch.inference_mode()
def encode_texts(texts: List[str], model, tokenizer, device: torch.device) -> torch.Tensor:
    """Encode a list of texts into normalized CLIP text embeddings."""
    tokens = tokenizer(texts)
    tokens = {k: v.to(device) for k, v in tokens.items()} if isinstance(tokens, dict) else tokens.to(device)
    feats = model.encode_text(tokens)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats

@torch.inference_mode()
def compute_class_embeddings(classes: List[str], model, tokenizer, device: torch.device) -> torch.Tensor:
    """
    Compute averaged embeddings per class across templates.
    Returns: [num_classes, dim]
    """
    prompts = []
    idx_slices = []  # start,end indices per class in flat prompt list
    start = 0
    for c in classes:
        ps = _templ(c)
        prompts += ps
        end = start + len(ps)
        idx_slices.append((start, end))
        start = end

    feats = encode_texts(prompts, model, tokenizer, device)              # [sum_T, dim]
    out = []
    for s, e in idx_slices:
        out.append(feats[s:e].mean(dim=0, keepdim=True))                  # [1, dim]
    return torch.cat(out, dim=0)                                          # [C, dim]

def _topk_with_sims(caption: str,
                    class_embs: torch.Tensor,
                    model, tokenizer, device: torch.device,
                    k: int) -> Tuple[List[int], List[float], torch.Tensor, torch.Tensor]:
    """
    Return:
      - topk indices and scores for classes,
      - cap_feat (1, dim),
      - full class sims (C,)
    """
    cap_feat = encode_texts([caption], model, tokenizer, device)          # [1, dim]
    sims = (cap_feat @ class_embs.T).squeeze(0)                           # [C]
    vals, idx = torch.topk(sims, k=min(k, class_embs.size(0)))
    return idx.tolist(), vals.tolist(), cap_feat, sims

@torch.inference_mode()
def _encode_attack_word_avg(word: str,
                            model, tokenizer, device: torch.device) -> torch.Tensor:
    """Encode an attack word with the same templates, average to a single vector [dim]."""
    prompts = _templ(word)
    emb = encode_texts(prompts, model, tokenizer, device)                 # [T, dim]
    return emb.mean(dim=0)                                                # [dim]


def clip_match(
        classes: List[str],
        examples: List[Dict[str, Any]],
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        attack_key: str = "attack_word",
    ) -> Dict[str, Any]:
    """
    Compute:
      acc_at_1, acc_at_5 : top-1 / top-5 accuracy
      asr_at_1, asr_at_5 : top-1 / top-5 attack success rate
    Each example can optionally include an `attack_key` (e.g., "attack_word").
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)

    class_embs = compute_class_embeddings(classes, model, tokenizer, device)
    name2id = {name: i for i, name in enumerate(classes)}
    C = class_embs.size(0)

    attack_cache: Dict[str, torch.Tensor] = {}
    per_example = []

    hit1 = hit5 = 0
    atk1 = atk5 = 0
    num_examples = len(examples)
    num_with_attack = 0

    for ex in examples:
        image_id = ex["image_id"]
        question_id = ex.get("question_id", None)
        gt_labels = ex["gt_labels"]
        caption = ex["pred_text"]
        attack_word: Optional[str] = ex.get(attack_key, None)

        # Compute similarities and Top-5 predictions
        _, _, cap_feat, sims = _topk_with_sims(caption, class_embs, model, tokenizer, device, k=5)
        vals5, idx5 = torch.topk(sims, k=min(5, C))
        top1_idx = int(idx5[0].item())
        top5_set = set(idx5.tolist())
        top1_pred = classes[top1_idx]
        top5_preds = [classes[i] for i in idx5.tolist()]

        gt_ids = {name2id[g] for g in gt_labels if g in name2id}

        # acc@1 / acc@5
        hit_at1 = any(top1_idx == gid for gid in gt_ids)
        hit_at5 = any(i in gt_ids for i in top5_set)
        hit1 += int(hit_at1)
        hit5 += int(hit_at5)

        # Attack success (if attack_word exists)
        attack_success_top1 = None
        attack_success_top5 = None
        if attack_word:
            num_with_attack += 1
            max_sim = float(vals5[0].item())
            kth5_sim = float(vals5[-1].item())
            if attack_word in name2id:
                aid = name2id[attack_word]
                attack_success_top1 = (aid == top1_idx)
                attack_success_top5 = (aid in top5_set)
            else:
                if attack_word not in attack_cache:
                    attack_cache[attack_word] = _encode_attack_word_avg(attack_word, model, tokenizer, device)
                atk_emb = attack_cache[attack_word]
                atk_sim = float((cap_feat @ atk_emb).item())
                attack_success_top1 = (atk_sim >= max_sim)
                attack_success_top5 = (atk_sim >= kth5_sim)
            atk1 += int(attack_success_top1)
            atk5 += int(attack_success_top5)

        per_example.append({
            "image_id": image_id,
            "question_id": question_id,
            "gt_labels": gt_labels,
            "pred_text": caption,
            "top1_pred": top1_pred,
            "top5_preds": top5_preds,
            "hit_at_1": hit_at1,
            "hit_at_5": hit_at5,
            "attack_word": attack_word,
            "attack_success_top1": attack_success_top1,
            "attack_success_top5": attack_success_top5,
        })

    # Compute aggregate metrics
    acc_at_1 = hit1 / num_examples if num_examples else 0.0
    acc_at_5 = hit5 / num_examples if num_examples else 0.0
    asr_at_1 = atk1 / num_with_attack if num_with_attack else None
    asr_at_5 = atk5 / num_with_attack if num_with_attack else None

    print(f"CLIP-Match: acc@1={acc_at_1:.4f}, acc@5={acc_at_5:.4f}, asr@1={asr_at_1}, asr@5={asr_at_5} (num_examples={num_examples}, with_attack={num_with_attack})")

    return {
        "acc_at_1": acc_at_1,
        "acc_at_5": acc_at_5,
        "asr_at_1": asr_at_1,
        "asr_at_5": asr_at_5,
        "num_examples": num_examples,
        "num_examples_with_attack": num_with_attack,
        "records": per_example,
    }


if __name__ == "__main__":
    import pprint

    classes = ["Bottle", "Person", "Cellphone", "TV", "Dog", "Cat", "Elephant", "Sofa", "Table"]
    examples = [
        {
            "image_id": "img1",
            "gt_labels": ["Bottle", "Person"],
            "pred_text": "a person holding a bottle next to a tv",
            "attack_word": "Cat",          # in-classes negative
        },
        {
            "image_id": "img2",
            "gt_labels": ["Dog"],
            "pred_text": "a cat sitting on a sofa",
            "attack_word": "Elephant",     # external negative
        },
        {
            "image_id": "img3",
            "gt_labels": ["TV"],
            "pred_text": "a cellphone on the table",
            # no attack_word provided
        },
        {
            "image_id": "img4",
            "gt_labels": ["Dog", "Cat"],
            "pred_text": "a dog and a cat playing together",
            "attack_word": "Sofa",          # in-classes positive
        },
        {
            "image_id": "img5",
            "gt_labels": ["Elephant"],
            "pred_text": "an elephant walking in the wild",
            "attack_word": "Cat",       # external positive
        },
    ]

    out = clip_match(
        classes, examples,
        model_name="ViT-B-32", pretrained="openai",
        attack_key="attack_word",
    )
    pprint.pprint(out)
