# RIO-Bench Data Construction

This document describes the data construction pipeline for **RIO-Bench**.  
For evaluation procedures, please refer to the main [README](../README.md).

This directory contains scripts used to construct the four RIO-Bench datasets:
- **Obj-Clean**
- **Obj-Attack**
- **Text-Clean**
- **Text-Attack**

We release the full construction pipeline to ensure **transparency**, **reproducibility**, and **extendability**.  
Researchers may adapt this pipeline for:
- generating improved training data,
- extending RIO-Bench,
- or building new robustness benchmarks.


### RIO-Bench Construction Overview
<img src="./figures/pipeline_light.png" alt="RIO-Bench Construction Overview" width="600"/>


# A. Object-VQA (Obj-Clean, Obj-Attack)

Before running any scripts, navigate to the `data_construction` directory:
```bash
cd data_construction
```

## A.1. Preparation

### A.1.1. Download assets

**Open Image's class hierarchy file**
- Download: https://github.com/openimages/dataset/blob/main/assets/bbox_hierarchy.json
- Place the file under: `assets/open_images/`

**TextVQA OCR results**
- Download "Rosetta OCR tokens [v0.2]" (train and val splits) from: https://textvqa.org/dataset/
- Place the file under: `assets/textvqa_meta/`

### A.1.2. Build hierarchy data structure + (optional) visualize trees
```bash
python3 prepare_metadata/build_hierarchy_maps.py
python3 prepare_metadata/print_hierarchy_trees.py
```
These hierarchy maps are used to:
- prune labels,
- generate multiple-choice options,
- and construct attack text candidates.

### A.1.3. Calculate CLIP scores for all labels
```bash
python3 obj_get_reliable_class.py
```
Open Images V3 assigns multiple labels per image without explicit priority.
- We compute CLIP similarity between image embeddings and each label’s text embedding.
- The scores are stored and later used to:
  - determine pseudo-ground-truth labels,
  - generate harder / easier distractors.
  
## A.2. Multiple-choice generation
```bash
python3 obj_create_multi_choice_data.py
```
This script performs:
- **1. Ground-truth label selection**
  - via hierarchy pruning + CLIP scoring.
- **2. Negative label generation (hard / medium / easy)**
  - based on semantic proximity in the hierarchy.

## A.3. Typographic attack generation
```bash
python3 obj_typo_attack.py
```
- Generates adversarial examples by inserting misleading text into images.
- Text placement respects OCR bounding boxes to avoid collisions with existing text.
- Three difficulty levels:
  - Hard: top-1 most confusing distractor
  - Medium: randomly sampled medium-level distractor
  - Easy: randomly sampled easy distractor
- Outputs are saved in structured directories for each configuration.

## A.4. Create datasets: generate Q&A (MC/OE) and combine with attacked images
```python3 obj_create_dataset.py```
This script:
- Combines the generated multiple-choice questions and typographic attacks with:
  - clean images,
  - or attacked images.
- Produces the final Obj-Clean and Obj-Attack HuggingFace datasets, ready for evaluation.

# B. Text-VQA (Text-Clean, Text-Attack)

## B.1. Generate attack words using LLM
```python3 txt_gen_misleading_word.py```
- Uses Llama-3.1 to generate contextually plausible but incorrect words/phrases.
- These serve as adversarial text tokens for typographic attacks in the Text-VQA setting.

## B.2. Typographic attack generation
```python3 txt_typo_attack.py```
- Inserts misleading text into TextVQA images using the generated words.
- Uses OCR bounding boxes to ensure non-overlapping placement with existing text.
- Attack levels:
  - Hard: placed near the key text regions
  - Easy: placed farther away from key text regions
- Outputs both images and detailed metadata. 

## B.3. Create datasets
```python3 txt_create_dataset.py```
- Produces HuggingFace datasets for:
  - Text-Clean
  - Text-Attack
- Includes attack metadata inside each dataset entry for detailed analysis and ablation.

# C. Filter Dataset to Unique Images
```python3 filter_unique_image.py```
- Removes duplicate entries so that each image appears exactly once in the dataset.
- This is required because:
  - TextVQA has multiple questions per image,
  - whereas Obj-VQA is defined with a single object-centric question per image.
- When mapping TextVQA images into the Obj-VQA format, this mismatch leads to multiple Obj-VQA-style samples for the same image.
- The script deduplicates these samples, enforcing the one-question-per-image structure expected in Obj-VQA.