#!/bin/bash

model="unsloth/llava-1.5-7b-hf-bnb-4bit"
model_name="unsloth-llava-1.5-7b-hf-bnb-4bit"

REPO_ID="${REPO_ID:-turing-motors/RIO-Bench}"
HF_TOKEN="${HF_TOKEN:-}"

dataset1="train/obj_attack__mc_hard"
dataset2="train/obj_attack__oe_hard"
dataset3="train/txt_attack__oe_hard"
dataset_source="hf:${REPO_ID}"
ds_num1=4000
ds_num2=4000
ds_num3=8000

dataname="oa-mch${ds_num1}-oeh${ds_num2}-ta-oem${ds_num3}"
output_dir="./outputs/ckpts/${dataname}/${model_name}/SFT_example"

target_layers="vision,multi_modal_projector,language"
seed=42

echo "===== SFT Training Start ====="
echo "Model:        ${model}"
echo "Data (train):"
echo "  - ${dataset1} (${ds_num1})"
echo "  - ${dataset2} (${ds_num2})"
echo "  - ${dataset3} (${ds_num3})"
echo "Output dir:   ${output_dir}"
echo "Data source:  ${dataset_source}"
echo "Target layers:${target_layers}"
echo "Seed:         ${seed}"
echo "==============================="

# ===== Train (single run) =====
python3 -m src.train_sft_unsloth \
    --model_name_or_path "${model}" \
    --dataset_name "${dataset1}" "${dataset2}" "${dataset3}" \
    --dataset_sample_num ${ds_num1} ${ds_num2} ${ds_num3} \
    --repo_id "${REPO_ID}" \
    --hf_token "${HF_TOKEN}" \
    --data_root "${DATA_ROOT}" \
    --output_dir "${output_dir}" \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.02 \
    --target_layers "${target_layers}" \
    --lora_r 16 \
    --lora_alpha 16 \
    --finetune_vision_layers true \
    --finetune_language_layers true \
    --finetune_attention_modules true \
    --finetune_mlp_modules true \
    --seed ${seed}
