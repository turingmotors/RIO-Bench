#!/bin/bash

model=llava-hf/llava-1.5-13b-hf
model_name=llava-1.5-13b-hf

REPO_ID="${REPO_ID:-turing-motors/RIO-Bench}"
HF_TOKEN="${HF_TOKEN:-}"

dataset1="train/obj_attack__mc_hard"
dataset2="train/obj_attack__oe_hard"
dataset3="train/txt_attack__oe_hard"
dataset_source="hf:${REPO_ID}"


ds_num1=4000
ds_num2=4000
ds_num3=8000

ds_num1=200
ds_num2=100
ds_num3=100


dataname="oa-mch${ds_num1}-oeh${ds_num2}-ta-oem${ds_num3}"
output_dir="./outputs/ckpts/${dataname}/${model_name}/SFT_example"


# MllamaForConditionalGeneration(
#   (model): MllamaModel(
#     (vision_model): MllamaVisionModel(
#       (patch_embedding): Conv2d(3, 1280, kernel_size=(14, 14), stride=(14, 14), padding=valid, bias=False)
#       (gated_positional_embedding): MllamaPrecomputedPositionEmbedding(
#         (tile_embedding): Embedding(9, 8197120)
#       )
#       (pre_tile_positional_embedding): MllamaPrecomputedAspectRatioEmbedding(
#         (embedding): Embedding(9, 5120)
#       )
#       (post_tile_positional_embedding): MllamaPrecomputedAspectRatioEmbedding(
#         (embedding): Embedding(9, 5120)
#       )
#       (layernorm_pre): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#       (layernorm_post): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#       (transformer): MllamaVisionEncoder(
#         (layers): ModuleList(
#           (0-31): 32 x MllamaVisionEncoderLayer(
#             (self_attn): MllamaVisionAttention(
#               (q_proj): Linear4bit(in_features=1280, out_features=1280, bias=False)
#               (k_proj): Linear4bit(in_features=1280, out_features=1280, bias=False)
#               (v_proj): Linear4bit(in_features=1280, out_features=1280, bias=False)
#               (o_proj): Linear4bit(in_features=1280, out_features=1280, bias=False)
#             )
#             (mlp): MllamaVisionMLP(
#               (activation_fn): GELUActivation()
#               (fc1): Linear4bit(in_features=1280, out_features=5120, bias=True)
#               (fc2): Linear4bit(in_features=5120, out_features=1280, bias=True)
#             )
#             (input_layernorm): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#             (post_attention_layernorm): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#           )
#         )
#       )
#       (global_transformer): MllamaVisionEncoder(
#         (layers): ModuleList(
#           (0-7): 8 x MllamaVisionEncoderLayer(
#             (self_attn): MllamaVisionAttention(
#               (q_proj): Linear4bit(in_features=1280, out_features=1280, bias=False)
#               (k_proj): Linear4bit(in_features=1280, out_features=1280, bias=False)
#               (v_proj): Linear4bit(in_features=1280, out_features=1280, bias=False)
#               (o_proj): Linear4bit(in_features=1280, out_features=1280, bias=False)
#             )
#             (mlp): MllamaVisionMLP(
#               (activation_fn): GELUActivation()
#               (fc1): Linear4bit(in_features=1280, out_features=5120, bias=True)
#               (fc2): Linear4bit(in_features=5120, out_features=1280, bias=True)
#             )
#             (input_layernorm): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#             (post_attention_layernorm): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#           )
#         )
#       )
#     )
#     (language_model): MllamaTextModel(
#       (embed_tokens): Embedding(128264, 4096, padding_idx=128004)
#       (layers): ModuleList(
#         (0-2): 3 x MllamaSelfAttentionDecoderLayer(
#           (self_attn): MllamaTextSelfAttention(
#             (q_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (k_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (v_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (o_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#           )
#           (mlp): MllamaTextMLP(
#             (gate_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (up_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (down_proj): Linear4bit(in_features=14336, out_features=4096, bias=False)
#             (act_fn): SiLU()
#           )
#           (input_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#           (post_attention_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#         )
#         (3): MllamaCrossAttentionDecoderLayer(
#           (cross_attn): MllamaTextCrossAttention(
#             (q_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (k_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (v_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (o_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (q_norm): MllamaTextRMSNorm((128,), eps=1e-05)
#             (k_norm): MllamaTextRMSNorm((128,), eps=1e-05)
#           )
#           (input_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#           (mlp): MllamaTextMLP(
#             (gate_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (up_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (down_proj): Linear4bit(in_features=14336, out_features=4096, bias=False)
#             (act_fn): SiLU()
#           )
#           (post_attention_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#         )
#         (4-7): 4 x MllamaSelfAttentionDecoderLayer(
#           (self_attn): MllamaTextSelfAttention(
#             (q_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (k_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (v_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (o_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#           )
#           (mlp): MllamaTextMLP(
#             (gate_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (up_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (down_proj): Linear4bit(in_features=14336, out_features=4096, bias=False)
#             (act_fn): SiLU()
#           )
#           (input_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#           (post_attention_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#         )
#         (8): MllamaCrossAttentionDecoderLayer(
#           (cross_attn): MllamaTextCrossAttention(
#             (q_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (k_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (v_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (o_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (q_norm): MllamaTextRMSNorm((128,), eps=1e-05)
#             (k_norm): MllamaTextRMSNorm((128,), eps=1e-05)
#           )
#           (input_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#           (mlp): MllamaTextMLP(
#             (gate_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (up_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (down_proj): Linear4bit(in_features=14336, out_features=4096, bias=False)
#             (act_fn): SiLU()
#           )
#           (post_attention_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#         )
#         (9-12): 4 x MllamaSelfAttentionDecoderLayer(
#           (self_attn): MllamaTextSelfAttention(
#             (q_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (k_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (v_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (o_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#           )
#           (mlp): MllamaTextMLP(
#             (gate_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (up_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (down_proj): Linear4bit(in_features=14336, out_features=4096, bias=False)
#             (act_fn): SiLU()
#           )
#           (input_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#           (post_attention_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#         )
#         (13): MllamaCrossAttentionDecoderLayer(
#           (cross_attn): MllamaTextCrossAttention(
#             (q_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (k_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (v_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (o_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (q_norm): MllamaTextRMSNorm((128,), eps=1e-05)
#             (k_norm): MllamaTextRMSNorm((128,), eps=1e-05)
#           )
#           (input_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#           (mlp): MllamaTextMLP(
#             (gate_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (up_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (down_proj): Linear4bit(in_features=14336, out_features=4096, bias=False)
#             (act_fn): SiLU()
#           )
#           (post_attention_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#         )
#         (14-17): 4 x MllamaSelfAttentionDecoderLayer(
#           (self_attn): MllamaTextSelfAttention(
#             (q_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (k_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (v_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (o_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#           )
#           (mlp): MllamaTextMLP(
#             (gate_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (up_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (down_proj): Linear4bit(in_features=14336, out_features=4096, bias=False)
#             (act_fn): SiLU()
#           )
#           (input_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#           (post_attention_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#         )
#         (18): MllamaCrossAttentionDecoderLayer(
#           (cross_attn): MllamaTextCrossAttention(
#             (q_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (k_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (v_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (o_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (q_norm): MllamaTextRMSNorm((128,), eps=1e-05)
#             (k_norm): MllamaTextRMSNorm((128,), eps=1e-05)
#           )
#           (input_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#           (mlp): MllamaTextMLP(
#             (gate_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (up_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (down_proj): Linear4bit(in_features=14336, out_features=4096, bias=False)
#             (act_fn): SiLU()
#           )
#           (post_attention_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#         )
#         (19-22): 4 x MllamaSelfAttentionDecoderLayer(
#           (self_attn): MllamaTextSelfAttention(
#             (q_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (k_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (v_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (o_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#           )
#           (mlp): MllamaTextMLP(
#             (gate_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (up_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (down_proj): Linear4bit(in_features=14336, out_features=4096, bias=False)
#             (act_fn): SiLU()
#           )
#           (input_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#           (post_attention_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#         )
#         (23): MllamaCrossAttentionDecoderLayer(
#           (cross_attn): MllamaTextCrossAttention(
#             (q_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (k_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (v_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (o_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (q_norm): MllamaTextRMSNorm((128,), eps=1e-05)
#             (k_norm): MllamaTextRMSNorm((128,), eps=1e-05)
#           )
#           (input_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#           (mlp): MllamaTextMLP(
#             (gate_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (up_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (down_proj): Linear4bit(in_features=14336, out_features=4096, bias=False)
#             (act_fn): SiLU()
#           )
#           (post_attention_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#         )
#         (24-27): 4 x MllamaSelfAttentionDecoderLayer(
#           (self_attn): MllamaTextSelfAttention(
#             (q_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (k_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (v_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (o_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#           )
#           (mlp): MllamaTextMLP(
#             (gate_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (up_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (down_proj): Linear4bit(in_features=14336, out_features=4096, bias=False)
#             (act_fn): SiLU()
#           )
#           (input_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#           (post_attention_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#         )
#         (28): MllamaCrossAttentionDecoderLayer(
#           (cross_attn): MllamaTextCrossAttention(
#             (q_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (k_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (v_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (o_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (q_norm): MllamaTextRMSNorm((128,), eps=1e-05)
#             (k_norm): MllamaTextRMSNorm((128,), eps=1e-05)
#           )
#           (input_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#           (mlp): MllamaTextMLP(
#             (gate_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (up_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (down_proj): Linear4bit(in_features=14336, out_features=4096, bias=False)
#             (act_fn): SiLU()
#           )
#           (post_attention_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#         )
#         (29-32): 4 x MllamaSelfAttentionDecoderLayer(
#           (self_attn): MllamaTextSelfAttention(
#             (q_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (k_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (v_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (o_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#           )
#           (mlp): MllamaTextMLP(
#             (gate_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (up_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (down_proj): Linear4bit(in_features=14336, out_features=4096, bias=False)
#             (act_fn): SiLU()
#           )
#           (input_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#           (post_attention_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#         )
#         (33): MllamaCrossAttentionDecoderLayer(
#           (cross_attn): MllamaTextCrossAttention(
#             (q_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (k_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (v_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (o_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (q_norm): MllamaTextRMSNorm((128,), eps=1e-05)
#             (k_norm): MllamaTextRMSNorm((128,), eps=1e-05)
#           )
#           (input_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#           (mlp): MllamaTextMLP(
#             (gate_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (up_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (down_proj): Linear4bit(in_features=14336, out_features=4096, bias=False)
#             (act_fn): SiLU()
#           )
#           (post_attention_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#         )
#         (34-37): 4 x MllamaSelfAttentionDecoderLayer(
#           (self_attn): MllamaTextSelfAttention(
#             (q_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (k_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (v_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (o_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#           )
#           (mlp): MllamaTextMLP(
#             (gate_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (up_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (down_proj): Linear4bit(in_features=14336, out_features=4096, bias=False)
#             (act_fn): SiLU()
#           )
#           (input_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#           (post_attention_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#         )
#         (38): MllamaCrossAttentionDecoderLayer(
#           (cross_attn): MllamaTextCrossAttention(
#             (q_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (k_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (v_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (o_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (q_norm): MllamaTextRMSNorm((128,), eps=1e-05)
#             (k_norm): MllamaTextRMSNorm((128,), eps=1e-05)
#           )
#           (input_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#           (mlp): MllamaTextMLP(
#             (gate_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (up_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (down_proj): Linear4bit(in_features=14336, out_features=4096, bias=False)
#             (act_fn): SiLU()
#           )
#           (post_attention_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#         )
#         (39): MllamaSelfAttentionDecoderLayer(
#           (self_attn): MllamaTextSelfAttention(
#             (q_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#             (k_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (v_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#             (o_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#           )
#           (mlp): MllamaTextMLP(
#             (gate_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (up_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#             (down_proj): Linear4bit(in_features=14336, out_features=4096, bias=False)
#             (act_fn): SiLU()
#           )
#           (input_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#           (post_attention_layernorm): MllamaTextRMSNorm((4096,), eps=1e-05)
#         )
#       )
#       (norm): MllamaTextRMSNorm((4096,), eps=1e-05)
#       (rotary_emb): MllamaRotaryEmbedding()
#     )
#     (multi_modal_projector): Linear4bit(in_features=7680, out_features=4096, bias=True)
#   )
#   (lm_head): Linear(in_features=4096, out_features=128256, bias=False)
# )

target_layers="vision_model,language_model,multi_modal_projector"
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
python3 -m src.train_sft_peft \
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
    --seed ${seed} \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8
