

python scripts/rebuild_dataset_from_images.py \
  --attack_dataset_dir ./scenetap_hf/obj_attack__mc_hard__scenetap/val \
  --images_dir ./logs/gpt-4o/val__obj_attack__mc_hard/SceneTAP/plan_llm/slider_3.0/filter_12.0/merged_images \
  --out_root ./scenetap_hf \
  --suffix <run_id> \
  --prefer_question_id