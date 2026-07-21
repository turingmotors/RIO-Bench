# SceneTAP: Scene-Coherent Typographic Adversarial Planner Against Vision-Language Models in Real-World Environments

> This directory is a modified copy of the official [SceneTAP](https://github.com/tsingqguo/scenetap)
> implementation, adapted to generate the `*__scenetap` attack variants of RIO-Bench. All credit for
> the original method goes to the SceneTAP authors (see Citation below); please cite their paper if you
> use this code.

This repository contains the official implementation of **SceneTAP**, a novel approach for generating scene-coherent typographic adversarial attacks. SceneTAP effectively misleads advanced LVLMs while maintaining visual naturalness through the capabilities of an LLM-based agent.

This work has been accepted to **CVPR 2025**.

![fig1](figs/fig1.png)

## Environment Setup

### Step 1: Configure the SoM Environment

Follow the setup instructions in the [SoM repository](https://github.com/microsoft/SoM) to configure the SoM environment.

### Step 2: Configure the TextDiffuser-2 Environment

Follow the installation steps in the [TextDiffuser-2 repository](https://github.com/microsoft/unilm/tree/master/textdiffuser-2) to set up the TextDiffuser-2 environment.

## Implementation

### Step 1: Extract SoM Masks and Images

Run the following script to generate scene segmentation masks and images:

```bash
python save_som_images.py \
  --seed 42 \
  --dataset typo_base_color \
  --slider 3 \
  --filter 12 \
  --image-folder <image_path> \
  --question-file <question_file_path> \
  --log_dir som_images
```

For further details, refer to `scripts/save_som_images.sh`.

### Step 2: Run the SceneTAP Attack

Once the SoM images are generated, execute the SceneTAP attack using:

```bash
python chatgpt_test.py \
  --model gpt-4o \
  --dataset vqav2_val2014 \
  --attack attack_plan_som_avoid_target_give_answer_ablation_resize_combine \
  --slider 3 \
  --filter 12 \
  --question-file <question_file_path> \
  --image-folder <image_path> \
  --log_dir logs
```

For additional details, refer to `scripts/chatgpt_test.sh` (RIO-Bench obj-attack: `mc_hard__scenetap`)
and `scripts/chatgpt_test_text_attack.sh` (RIO-Bench txt-attack: `oe_hard__scenetap`), which differ only
in `--dataset_name` / `--clean_base_dataset`.

### Step 3: Derive the Obj-Attack Open-Ended Variant

`chatgpt_test.sh` produces the multiple-choice `obj_attack__mc_hard__scenetap` dataset. To reuse the
same attacked images for the open-ended variant (`obj_attack__oe_hard__scenetap`), run:

```bash
python scripts/build_obj_oe_from_scenetap_mc.py \
  --input_mc_dataset ./scenetap_hf/<run_id>/val/obj_attack__mc_hard__scenetap \
  --output_oe_dataset ./scenetap_hf/<run_id>/val/obj_attack__oe_hard__scenetap \
  --split validation
```

This does not call the LLM again -- it relabels each MC sample's question/answer as open-ended
(pruning ground-truth object classes to pseudo-leaves in the Open Images hierarchy, same as
`obj_create_dataset.py`'s OE generation) while keeping the SceneTAP-attacked image unchanged.
There is no open-ended counterpart for `txt_attack__oe_hard__scenetap` since the text-attack path
is already open-ended.

### Data Format

The question file follows a JSON format and contains multiple entries:

```
[
  {
    "question_id": <question_id>,
    "image": "<image_name>",
    "text": "<question>",
    "category": "<category>",
    "answer": "<answer>"
  },
  {
    "question_id": <question_id>,
    "image": "<image_name>",
    "text": "<question>",
    "category": "<category>",
    "answer": "<answer>"
  },
  ...
]
```

## Citation

If you use this code, please cite our CVPR 2025 paper:

```
@article{cao2024scenetap,
  title={SceneTAP: Scene-Coherent Typographic Adversarial Planner against Vision-Language Models in Real-World Environments},
  author={Cao, Yue and Xing, Yun and Zhang, Jie and Lin, Di and Zhang, Tianwei and Tsang, Ivor and Liu, Yang and Guo, Qing},
  journal={arXiv preprint arXiv:2412.00114},
  year={2024}
}
```