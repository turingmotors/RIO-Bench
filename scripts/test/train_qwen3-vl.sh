#!/bin/bash
#SBATCH --job-name=TypoDefense_v1_SFT
#SBATCH --time=24:00:00
#SBATCH --partition=h100
#SBATCH --nodes 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.out
#SBATCH --array=0

# Init: login to huggingface and wandb, load modules, activate venv
cd /home/futa_waseda/VLMdefense
. init.sh


# model="llava-hf/llava-1.5-7b-hf"
# model_name=llava-1.5-7b-hf
model="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
model_name=unsloth-qwen3-vl-8b-instruct-unsloth-bnb-4bit

base_dir=/home/futa_waseda/VLMdefense/data/RIO-Bench/hf_dataset_unique_img/train
tr_dataset_list=(
    "${base_dir}/obj_attack/mcq_random_hard"
    "${base_dir}/obj_attack/open_ended_random_hard_v3"
    "${base_dir}/txt_attack/open_ended_mid"
)
ds_num_list=(
    4000
    4000
    8000
)
dataname="oa-mcqh${ds_num_list[0]}-oehv3${ds_num_list[1]}-ta-oem${ds_num_list[2]}"

epoch_list=(1)
lora_r_list=(16)
lora_alpha_list=(16)
lr_list=(1e-4)


# Qwen3VLForConditionalGeneration(
#   (model): Qwen3VLModel(
#     (visual): Qwen3VLVisionModel(
#       (patch_embed): Qwen3VLVisionPatchEmbed(
#         (proj): Conv3d(3, 1152, kernel_size=(2, 16, 16), stride=(2, 16, 16))
#       )
#       (pos_embed): Embedding(2304, 1152)
#       (rotary_pos_emb): Qwen3VLVisionRotaryEmbedding()
#       (blocks): ModuleList(
#         (0-26): 27 x Qwen3VLVisionBlock(
#           (norm1): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
#           (norm2): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
#           (attn): Qwen3VLVisionAttention(
#             (qkv): Linear(in_features=1152, out_features=3456, bias=True)
#             (proj): Linear(in_features=1152, out_features=1152, bias=True)
#           )
#           (mlp): Qwen3VLVisionMLP(
#             (linear_fc1): Linear(in_features=1152, out_features=4304, bias=True)
#             (linear_fc2): Linear(in_features=4304, out_features=1152, bias=True)
#             (act_fn): GELUTanh()
#           )
#         )
#       )
#       (merger): Qwen3VLVisionPatchMerger(
#         (norm): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
#         (linear_fc1): Linear(in_features=4608, out_features=4608, bias=True)
#         (act_fn): GELU(approximate='none')
#         (linear_fc2): Linear(in_features=4608, out_features=4096, bias=True)
#       )
#       (deepstack_merger_list): ModuleList(
#         (0-2): 3 x Qwen3VLVisionPatchMerger(
#           (norm): LayerNorm((4608,), eps=1e-06, elementwise_affine=True)
#           (linear_fc1): Linear(in_features=4608, out_features=4608, bias=True)
#           (act_fn): GELU(approximate='none')
#           (linear_fc2): Linear(in_features=4608, out_features=4096, bias=True)
#         )
#       )
#     )
#     (language_model): Qwen3VLTextModel(
#       (embed_tokens): Embedding(151936, 4096)
#       (layers): ModuleList(
#         (0-5): 6 x Qwen3VLTextDecoderLayer(
#           (self_attn): Qwen3VLTextAttention(
#             (q_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (k_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (v_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (o_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (q_norm): Qwen3VLTextRMSNorm((128,), eps=1e-06)
#             (k_norm): Qwen3VLTextRMSNorm((128,), eps=1e-06)
#           )
#           (mlp): Qwen3VLTextMLP(
#             (gate_proj): Linear4bit(in_features=4096, out_features=12288, bias=False)
#             (up_proj): Linear4bit(in_features=4096, out_features=12288, bias=False)
#             (down_proj): Linear4bit(in_features=12288, out_features=4096, bias=False)
#             (act_fn): SiLUActivation()
#           )
#           (input_layernorm): Qwen3VLTextRMSNorm((4096,), eps=1e-06)
#           (post_attention_layernorm): Qwen3VLTextRMSNorm((4096,), eps=1e-06)
#         )
#         (6): Qwen3VLTextDecoderLayer(
#           (self_attn): Qwen3VLTextAttention(
#             (q_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (k_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (v_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (o_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (q_norm): Qwen3VLTextRMSNorm((128,), eps=1e-06)
#             (k_norm): Qwen3VLTextRMSNorm((128,), eps=1e-06)
#           )
#           (mlp): Qwen3VLTextMLP(
#             (gate_proj): Linear(in_features=4096, out_features=12288, bias=False)
#             (up_proj): Linear(in_features=4096, out_features=12288, bias=False)
#             (down_proj): Linear(in_features=12288, out_features=4096, bias=False)
#             (act_fn): SiLUActivation()
#           )
#           (input_layernorm): Qwen3VLTextRMSNorm((4096,), eps=1e-06)
#           (post_attention_layernorm): Qwen3VLTextRMSNorm((4096,), eps=1e-06)
#         )
#         (7-15): 9 x Qwen3VLTextDecoderLayer(
#           (self_attn): Qwen3VLTextAttention(
#             (q_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (k_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (v_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (o_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (q_norm): Qwen3VLTextRMSNorm((128,), eps=1e-06)
#             (k_norm): Qwen3VLTextRMSNorm((128,), eps=1e-06)
#           )
#           (mlp): Qwen3VLTextMLP(
#             (gate_proj): Linear4bit(in_features=4096, out_features=12288, bias=False)
#             (up_proj): Linear4bit(in_features=4096, out_features=12288, bias=False)
#             (down_proj): Linear4bit(in_features=12288, out_features=4096, bias=False)
#             (act_fn): SiLUActivation()
#           )
#           (input_layernorm): Qwen3VLTextRMSNorm((4096,), eps=1e-06)
#           (post_attention_layernorm): Qwen3VLTextRMSNorm((4096,), eps=1e-06)
#         )
#         (16): Qwen3VLTextDecoderLayer(
#           (self_attn): Qwen3VLTextAttention(
#             (q_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (k_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (v_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
#             (q_norm): Qwen3VLTextRMSNorm((128,), eps=1e-06)
#             (k_norm): Qwen3VLTextRMSNorm((128,), eps=1e-06)
#           )
#           (mlp): Qwen3VLTextMLP(
#             (gate_proj): Linear4bit(in_features=4096, out_features=12288, bias=False)
#             (up_proj): Linear4bit(in_features=4096, out_features=12288, bias=False)
#             (down_proj): Linear4bit(in_features=12288, out_features=4096, bias=False)
#             (act_fn): SiLUActivation()
#           )
#           (input_layernorm): Qwen3VLTextRMSNorm((4096,), eps=1e-06)
#           (post_attention_layernorm): Qwen3VLTextRMSNorm((4096,), eps=1e-06)
#         )
#         (17-34): 18 x Qwen3VLTextDecoderLayer(
#           (self_attn): Qwen3VLTextAttention(
#             (q_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (k_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (v_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (o_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (q_norm): Qwen3VLTextRMSNorm((128,), eps=1e-06)
#             (k_norm): Qwen3VLTextRMSNorm((128,), eps=1e-06)
#           )
#           (mlp): Qwen3VLTextMLP(
#             (gate_proj): Linear4bit(in_features=4096, out_features=12288, bias=False)
#             (up_proj): Linear4bit(in_features=4096, out_features=12288, bias=False)
#             (down_proj): Linear4bit(in_features=12288, out_features=4096, bias=False)
#             (act_fn): SiLUActivation()
#           )
#           (input_layernorm): Qwen3VLTextRMSNorm((4096,), eps=1e-06)
#           (post_attention_layernorm): Qwen3VLTextRMSNorm((4096,), eps=1e-06)
#         )
#         (35): Qwen3VLTextDecoderLayer(
#           (self_attn): Qwen3VLTextAttention(
#             (q_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (k_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (v_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (o_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (q_norm): Qwen3VLTextRMSNorm((128,), eps=1e-06)
#             (k_norm): Qwen3VLTextRMSNorm((128,), eps=1e-06)
#           )
#           (mlp): Qwen3VLTextMLP(
#             (gate_proj): Linear(in_features=4096, out_features=12288, bias=False)
#             (up_proj): Linear(in_features=4096, out_features=12288, bias=False)
#             (down_proj): Linear(in_features=12288, out_features=4096, bias=False)
#             (act_fn): SiLUActivation()
#           )
#           (input_layernorm): Qwen3VLTextRMSNorm((4096,), eps=1e-06)
#           (post_attention_layernorm): Qwen3VLTextRMSNorm((4096,), eps=1e-06)
#         )
#       )
#       (norm): Qwen3VLTextRMSNorm((4096,), eps=1e-06)
#       (rotary_emb): Qwen3VLTextRotaryEmbedding()
#     )
#   )
#   (lm_head): Linear(in_features=4096, out_features=151936, bias=False)
# )



target_layers_list=(
    "visual"
    "visual,language_model"
    "language_model"
)
target_layers_str_list=(
    "V-proj"
    "VL-proj"
    "L"
)

finetune_vision_layers=true
finetune_language_layers=true
finetune_attention_modules=true
finetune_mlp_modules=true

seed=42

count=0
for ep in "${epoch_list[@]}"; do
    for i in "${!lora_r_list[@]}"; do
        lora_r=${lora_r_list[i]}
        lora_alpha=${lora_alpha_list[i]}

        for lr in "${lr_list[@]}"; do


            for j in "${!target_layers_list[@]}"; do
                target_layers=${target_layers_list[$j]}
                target_layers_str=${target_layers_str_list[$j]}



                train_config_name="SFT_r${lora_r}_a${lora_alpha}_ep${ep}_lr${lr//./_}_train-${target_layers_str}-seed${seed}"

                output_dir="./outputs/ckpts/${dataname}/${model_name}/${train_config_name}"
                
                echo "Starting training at $(date) with ep=${ep}, lora_r=${lora_r}, lora_alpha=${lora_alpha}, lr=${lr}, finetune_vision_layers=${finetune_vision_layers}"
                echo "Output directory: ${output_dir}"

                source ~/miniconda3/etc/profile.d/conda.sh
                conda activate latest
                python train_sft.py \
                    --model_name_or_path ${model} \
                    --dataset_name ${tr_dataset_list[0]} ${tr_dataset_list[1]} ${tr_dataset_list[2]} \
                    --dataset_sample_num ${ds_num_list[0]} ${ds_num_list[1]} ${ds_num_list[2]} \
                    --warmup_ratio 0.02 \
                    --num_train_epochs ${ep} \
                    --learning_rate ${lr} \
                    --weight_decay 0.1 \
                    --output_dir ${output_dir} \
                    --target_layers ${target_layers} \
                    --lora_r ${lora_r} \
                    --lora_alpha ${lora_alpha} \
                    --finetune_vision_layers ${finetune_vision_layers} \
                    --finetune_language_layers ${finetune_language_layers} \
                    --finetune_attention_modules ${finetune_attention_modules} \
                    --finetune_mlp_modules ${finetune_mlp_modules} \
                    --seed ${seed}


                # Eval Typo
                source ~/miniconda3/etc/profile.d/conda.sh
                conda activate latest
                model_name_trained=${output_dir}

                eval_output_dir=${output_dir}/eval_results
                mkdir -p ${eval_output_dir}

                echo "Starting typo evaluation at $(date) with model ${model} and output dir ${eval_output_dir}"

                # Typo tasks
                data_root="/home/futa_waseda/VLMdefense/data/RIO-Bench/hf_dataset_unique_img/"
                dataset_list=(
                    val/obj_clean/mcq_clean
                    val/obj_attack/mcq_random_easy
                    val/obj_attack/mcq_random_medium
                    val/obj_attack/mcq_random_hard
                    val/obj_clean/open_ended_clean_v3
                    val/obj_attack/open_ended_random_easy_v3
                    val/obj_attack/open_ended_random_medium_v3
                    val/obj_attack/open_ended_random_hard_v3
                    val/txt_clean/original_with_suffix
                    val/txt_attack/open_ended_far
                    val/txt_attack/open_ended_mid
                )
                prompt_strategy_list=(
                    "1phase-basic"
                    # "2phase-focus"
                )
                for eval_i in "${!prompt_strategy_list[@]}"; do
                    prompt_strategy=${prompt_strategy_list[$eval_i]}
                    for eval_j in "${!dataset_list[@]}"; do
                        dataset=${dataset_list[$eval_j]}

                        echo "Starting evaluating ${dataset} with ${prompt_strategy} at $(date)"
                        python3 evaluate_rio_bench.py \
                            --model_name $model_name_trained \
                            --data_root "$data_root" \
                            --dataset_name $dataset \
                            --output_dir $eval_output_dir \
                            --prompt_strategy $prompt_strategy
                        echo "Finished evaluating ${dataset} with ${prompt_strategy} at $(date)"
                    done
                    echo "Finished evaluating all dactasets with ${prompt_strategy} at $(date)"
                done
                echo "Finished typo evaluation at $(date) with model ${model} and output dir ${eval_output_dir}"

            done
        done
    done
done