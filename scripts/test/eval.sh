#!/bin/bash
#SBATCH --job-name=eval_rio
#SBATCH --time=2-00:00
#SBATCH --partition=h100
#SBATCH --nodes 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.out
#SBATCH --array=0

HF_TOKEN="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"
huggingface-cli login --token $HF_TOKEN

model_name_list=(
    llava-hf/llava-1.5-7b-hf
    # llava-hf/llava-1.5-13b-hf
    # Qwen/Qwen3-VL-8B-Instruct
    # Qwen/Qwen2.5-VL-7B-Instruct
    # meta-llama/Llama-3.2-11B-Vision-Instruct
    # HuggingFaceTB/SmolVLM-Instruct
)

# Typo tasks
data_root="./data/RIO-Bench/hf_dataset_unique_img/"
dataset_list=(
    val/obj_clean/mc_clean
    val/obj_attack/mc_random_easy
    val/obj_attack/mc_random_medium
    val/obj_attack/mc_random_hard
    val/obj_clean/oe_clean
    val/obj_attack/oe_random_easy
    val/obj_attack/oe_random_medium
    val/obj_attack/oe_random_hard
    val/txt_clean/original_with_suffix
    val/txt_attack/open_ended_far
    val/txt_attack/open_ended_mid
)
prompt_strategy_list=(
    "1phase-basic"
    # "2phase-focus"
)


count=0
for i in "${!model_name_list[@]}"; do
    model_name=${model_name_list[$i]}
    if [[ $model_name == "/home"* ]]; then
        # If model_name is a local path
        output_dir="${model_name}/eval_results"
    else
        # If model_name is a Hugging Face model
        output_dir="./outputs/${model_name}/eval_results"
    fi

    # output_dir=${model_name}/eval_results
    mkdir -p ${output_dir}

    # run with job id = $SLURM_ARRAY_TASK_ID
    if [ $SLURM_ARRAY_TASK_ID -ne $count ]; then
        count=$((count + 1))
        continue
    fi

    echo "Starting evaluating with model ${model_name} at $(date)"
    
    # eval typo
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate latest
    echo "--- Evaluating RIO-Bench ---"
    for j in "${!prompt_strategy_list[@]}"; do
        prompt_strategy=${prompt_strategy_list[$j]}

        for k in "${!dataset_list[@]}"; do
            dataset=${dataset_list[$k]}

            echo "Starting evaluating ${dataset} with ${prompt_strategy} at $(date)"
            
            python3 evaluate_rio_bench.py \
                --model_name $model_name \
                --data_root "$data_root" \
                --dataset_name $dataset \
                --output_dir $output_dir \
                --prompt_strategy $prompt_strategy
            echo "Finished evaluating ${dataset} with ${prompt_strategy} at $(date)"
        done
        echo "Finished evaluating all datasets with ${prompt_strategy} at $(date)"
    done

    echo "Finished evaluating all prompt strategies at $(date)"
done