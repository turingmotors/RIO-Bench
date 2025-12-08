# RIO-Bench Data Construction
This document describes the data construction process for RIO-Bench.
For evaluation, please refer to the main [README](../README.md).
This directory contains scripts to create the RIO-Bench datasets: Obj-Clean, Obj-Attack, Text-Clean, and Text-Attack.
We release this code for transparency and reproducibility, as well as to facilitate future research in this area.
For example, this dataset construction pipeline can be adapted to other datasets, for generating improved training data, or for creating new benchmarks.

# A. Object-VQA: Obj-Clean, Obj-Attack

Don't forget to navigate to the data_construction directory first:
```
cd data_construction
```

## 1. Preparation

### 1-1. Download assets

Open Image's class hierarchy file 
- Download from https://github.com/openimages/dataset/blob/main/assets/bbox_hierarchy.json
- Place it in the folder `assets/open_images/`

TextVQA OCR results
- Download "Rosetta OCR tokens [v0.2]" from https://textvqa.org/dataset/ (both train and val splits)
- Place it in the folder `assets/textvqa_meta/`

### 1-2. Build hierarchy data structure + (optional) visualize trees
```
python3 prepare_metadata/build_hierarchy_maps.py
python3 prepare_metadata/print_hierarchy_trees.py
```
This hierarchy data will be used for generating multiple-choice questions + attack texts.

### 1-3. Calculate CLIP scores for all labels
```python3 obj_get_reliable_class.py```
- Open Images V3 dataset has multiple labels per image, whose priority is unknown.
- We use CLIP to compute similarity scores between image features and text features of each label.
- We save the scores in a JSON file for later use.

## 2. Multiple-choice generation
```python3 obj_create_multiple_choices_data.py```
- (1) Ground-truth label selection, based on hierarchy pruning and CLIP scores
- (2) Negatives (hard / medium / easy) for MCQ & distractor generation, using hierarchy proximity

## 3. Typographic attack generation
```python3 obj_typographic_attack.py```
- Using the multiple-choice questions from the previous step, this script generates adversarial examples by adding misleading text to images.
- Texts are placed in non-overlapping positions to existing texts in the image, using bboxes from OCR results (TextOCR).
- Three levels of attacks: hard, medium, easy.
    - Hard: use the top-1 negative (most confusing).
    - Medium: use a random negative from the medium pool.
    - Easy: use a random negative from the easy pool.
- The generated adversarial examples are saved in a specified directory.

## 4. Create datasets: generate Q&A (MC/OE) and combine with attacked images
```python3 obj_create_datasets.py```
- This script combines the generated multiple-choice questions and adversarial examples to create final datasets: Obj-Clean and Obj-Attack.
- It ensures that each dataset is properly formatted and ready for evaluation.


# B. Text-VQA (Text-Clean, Text-Attack)

## 1. Generate attack words using LLM
```python3 txt_gen_misleading_word.py```
- Based on TextVQA's original Q&A, LLM (llama-3.1) generates misleading words for typographic attacks.

## 2. Typographic attack generation
```python3 txt_typo_attack.py```
- Using the misleading words generated in the previous step, this script creates adversarial examples by adding misleading text to images.
- Texts are placed in non-overlapping positions to existing texts in the image, using bounding boxes from OCR results (TextOCR).
- Three levels of attacks: hard, easy.
    - Hard: closely placed from the key text
- The generated adversarial examples are saved in a specified directory.

## 3. Create datasets
```python3 txt_create_dataset.py```