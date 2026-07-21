#!/bin/bash
set -euo pipefail

python scripts/rebuild_dataset_from_images.py \
  --attack_dataset_dir ../../data/RIO-Bench/hf_dataset_unique_img/val/txt_attack/open_ended_mid \
  --images_dir ./logs-eccv/gpt-4o/val__txt_attack__oe_hard/SceneTAP/plan_llm/slider_3.0/filter_12.0/merged/images \
  --out_root ./scenetap_hf/logs-eccv \
  --out_attack_dataset_dir ./scenetap_hf/logs-eccv/val/txt_attack__oe_hard__scenetap \
  --suffix merged \
  --prefer_question_id
