#!/bin/bash
#SBATCH --job-name=som
#SBATCH --time=24:00:00
#SBATCH --partition=h100
#SBATCH --nodes 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.out
#SBATCH --array=0


python save_som_images.py \
  --seed 42 \
  --dataset_name train/obj_clean__mc_clean \
  --repo_id turing-motors/RIO-Bench \
  --slider 3 \
  --filter 12 \
  --log_dir som_images