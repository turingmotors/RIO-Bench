#!/bin/bash
#SBATCH --job-name=som_txt_val
#SBATCH --time=24:00:00
#SBATCH --partition=h100
#SBATCH --nodes 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.out

set -euo pipefail

python save_som_images.py \
  --seed 42 \
  --dataset val__txt_clean__original_with_suffix \
  --dataset_name val/txt_clean/original_with_suffix \
  --data_root ../../data/RIO-Bench/hf_dataset \
  --slider 3 \
  --filter 12 \
  --log_dir som_images
