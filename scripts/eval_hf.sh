#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

REPO_ID="${REPO_ID:-turing-motors/RIO-Bench}"
HF_TOKEN="${HF_TOKEN:-}"

model_name_list=(
    llava-hf/llava-1.5-7b-hf
    # llava-hf/llava-1.5-13b-hf
    # Qwen/Qwen3-VL-8B-Instruct
    # Qwen/Qwen2.5-VL-7B-Instruct
    # meta-llama/Llama-3.2-11B-Vision-Instruct
    # HuggingFaceTB/SmolVLM-Instruct
    
)

dataset_list=(
    val/obj_clean__mc_clean
    val/obj_attack__mc_easy
    val/obj_attack__mc_medium
    val/obj_attack__mc_hard
    val/obj_clean__oe_clean
    val/obj_attack__oe_easy
    val/obj_attack__oe_medium
    val/obj_attack__oe_hard
    val/txt_clean__oe_clean
    val/txt_attack__oe_easy
    val/txt_attack__oe_hard
)

prompt_strategy_list=(
    "1phase-basic"
    # "2phase-focus"
)

for model_name in "${model_name_list[@]}"; do
    if [[ $model_name == "/home"* ]]; then
        output_dir="${model_name}/eval_results"
    else
        output_dir="./outputs/${model_name}/eval_results"
    fi
    mkdir -p "${output_dir}"

    echo "Starting evaluating with model ${model_name} at $(date)"
    echo "--- Evaluating RIO-Bench ---"
    for prompt_strategy in "${prompt_strategy_list[@]}"; do
        for dataset in "${dataset_list[@]}"; do
            echo "Starting evaluating ${dataset} with ${prompt_strategy} at $(date)"
            python3 -m src.evaluate_rio_bench \
                --model_name "${model_name}" \
                --dataset_name "${dataset}" \
                --repo_id "${REPO_ID}" \
                --hf_token "${HF_TOKEN}" \
                --output_dir "${output_dir}" \
                --prompt_strategy "${prompt_strategy}"
            echo "Finished evaluating ${dataset} with ${prompt_strategy} at $(date)"
        done
        echo "Finished evaluating all datasets with ${prompt_strategy} at $(date)"
    done
    echo "Finished evaluating all prompt strategies at $(date)"
done
