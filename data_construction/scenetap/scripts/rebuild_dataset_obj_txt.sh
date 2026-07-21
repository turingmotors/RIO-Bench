

python scripts/rebuild_dataset_from_images.py \
  --attack_dataset_dir ../../data/RIO-Bench/hf_dataset_unique_img/val/txt_attack/open_ended_mid \
  --other_dataset_dir ../../data/RIO-Bench/hf_dataset_unique_img/val/obj_clean/mc_clean \
  --other_dataset_dir ../../data/RIO-Bench/hf_dataset_unique_img/val/txt_clean/original_with_suffix \
  --other_dataset_dir ./scenetap_hf/obj_attack__mc_hard__scenetap/val \
  --images_dir ./logs/gpt-4o/val__txt_attack__oe_hard/SceneTAP/plan_llm/slider_3.0/filter_12.0/seed_42/images \
  --out_root ./scenetap_hf \
  --suffix <run_id> \
  --prefer_question_id