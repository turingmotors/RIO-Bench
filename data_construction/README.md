# Obj-Clean, Obj-Attack

1. Preparation
```open_image_assets/open_image_hierarchy.py```
- Open Images V3 dataset has a hierarchical structure of labels.
- We process the hierarchy file provided by Open Images for preparation.

2. Calculate CLIP scores for all labels
```obj_get_reliable_class.py```
- Open Images V3 dataset has multiple labels per image, whose priority is unknown.
- We use CLIP to compute similarity scores between image features and text features of each label.
- We save the scores in a JSON file for later use.

## A. Multiple-choice question generation
3. Create multiple-choice questions
```obj_create_multiple_choices_data.py```
- (1) Y_true selection (ground-truth label used for evaluation)
    - Parent–child pruning (pseudo-leaves). If a parent and its child both appear in the image GT set, drop the parent and keep the child. (Operate only within the GT set.)
    - Depth preference: use leaves if available; otherwise, use the deepest nodes.
        - If any remaining GT label has depth ≥ 3, restrict candidates to those.
        - Otherwise (all candidates at depth ≤ 2), use all remaining GT labels.
    - Score-based choice: among the candidates, pick the one with the highest CLIP score.
    - Fallback: If none of the GT candidates has a score entry, pick the highest-scoring label under the root found in all_scores; if none, pick the global maximum.
- (2) Negatives (hard / medium / easy) for MCQ & distractor generation
    - Exclusion set: Start from the global vocabulary and exclude: (i) the pruned GT set (pseudo-leaves), (ii) all ancestors and descendants of those GT labels, and (iii) the selected y_true itself. Ensure all picks are distinct.
    - Locate y_true’s parent level. Let L_p be the deepest parent level of y_true (i.e., one level above y_true’s own depth).
    - Pools by hierarchy proximity (level-only, simple & deterministic):
        - Hard = siblings: labels that share the same ancestor at level L_p as y_true (i.e., same parent), excluding any ancestor/descendant of the GT.
            - Optional: remove items with very high co-occurrence with y_true (above θ_co) to avoid near-positives.
        - Medium = one level higher: labels that share an ancestor at level L_p−1 (but are not siblings and not in the exclusion set).
        - Easy = two levels higher: labels that share an ancestor at level L_p−2 (and pass exclusions). If L_p−2 is unavailable, back off to different top-branch (depth=2) under the root.
    - Back-off policy. If any pool is empty, widen the level (hard→medium→easy) or finally draw from the remaining vocabulary after exclusions. Keep results unique.
    - (Sampling fairness: use random sampling with a fixed random seed.)

4. Typographic attack generation
```obj_typographic_attack.py```
- Using the multiple-choice questions from the previous step, this script generates adversarial examples by adding misleading text to images.
- Texts are placed in non-overlapping positions to existing texts in the image, using bboxes from OCR results (TextOCR).
- Three levels of attacks: hard, medium, easy.
    - Hard: use the top-1 negative (most confusing).
    - Medium: use a random negative from the medium pool.
    - Easy: use a random negative from the easy pool.
- The generated adversarial examples are saved in a specified directory.

5. Create datasets
```obj_create_datasets.py```
- This script combines the generated multiple-choice questions and adversarial examples to create final datasets: Obj-Clean and Obj-Attack.
- It ensures that each dataset is properly formatted and ready for evaluation.

## B. Open-ended question generation
3. obj_create_open_ended_data.py
    - This script generates open-ended questions for object recognition tasks.
    - Correct labels are selected by
        - Pruning parent labels from the GT set. If both a parent and its child appear in the GT set, drop the parent and keep the child.
        - Remove labels that have very low CLIP scores (smaller than 5% stats) to guard against noise.

4. obj_typographic_attack_oe.py
    - This script generates adversarial examples for open-ended questions by adding misleading text to images.


# TextVQA-Clean, TextVQA-Attack