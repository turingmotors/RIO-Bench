#!/bin/bash
#SBATCH --job-name=chatgpt_test
#SBATCH --time=24:00:00
#SBATCH --partition=h100
#SBATCH --nodes 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.out
#SBATCH --array=0

python chatgpt_test.py \
  --model gpt-4o \
  --dataset_name val/txt_attack__oe_hard \
  --clean_base_dataset val/txt_clean__oe_clean \
  --repo_id turing-motors/RIO-Bench \
  --attack SceneTAP \
  --slider 3 \
  --filter 12 \
  --log_dir logs \
  --use_attack_word \
  --skip_model_eval \
  --save_hf_dataset \
  --output_dataset_root ./scenetap_hf \
  --som_ref_dataset obj_clean__mc_clean
