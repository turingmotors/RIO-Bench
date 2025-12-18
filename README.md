# Read or Ignore? A Unified Benchmark for Typographic-Attack Robustness and Text Recognition in Vision-Language Models


[![arXiv](https://img.shields.io/badge/arXiv-2401.12345-b31b1b?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2512.11899)
[![Project](https://img.shields.io/badge/Project-Website-111111?style=flat-square)](https://turingmotors.github.io/rio-vqa/)
<!-- [![HF](https://img.shields.io/badge/Hugging%20Face-Datasets-FFD21E?style=flat-square&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/xxx/yyy) -->


Real-world VLMs must decide when to **read** text and when to **ignore** it, e.g., reading traffic signs but not being fooled by text-based attacks on objects.

We propose a unified benchmark, **RIO-Bench**, to evaluate both typographic-attack robustness and text recognition in VLMs through a novel task called **RIO-VQA**.

### Problem Settings: VLMs Must Adaptively Read or Ignore Texts
<img src="./figures/teaser_light.png" alt="RIO-VQA Overview" width="600"/>


### RIO-VQA's Task Taxonomy
<img src="./figures/taxonomy_light.png" alt="RIO-VQA Taxonomy" width="600"/>

# Environment Setup
```bash
conda create -n riobench python=3.11 -y
bash setup.sh
```
(Additional dependencies may be required depending on your model/environment.)

# RIO-Bench Evaluation
```bash
bash ./scripts/eval_template.sh
```

# RIO-RT (Read-or-Ignore Robust Training)
```bash
bash ./scripts/train_unsloth_template.sh
```
For peft training, please create environment for peft, and run:
```bash
bash ./scripts/train_peft_template.sh
```

# RIO-Bench Construction Reproduction / Customization
Details are provided in: ```data_construction/README.md```[./data_construction/README.md]

# Cite
```
@article{waseda2025read,
  title        = {Read or Ignore? A Unified Benchmark for Typographic-Attack Robustness and Text Recognition in Vision-Language Models},
  author       = {Waseda, Futa and Yamabe, Shojiro and Shiono, Daiki and Sasaki, Kento and Takahashi, Tsubasa},
  year         = {2025},
  eprint       = {2512.11899},
  archivePrefix= {arXiv},
  primaryClass = {cs.CV},
  url          = {https://arxiv.org/abs/2512.11899},
}
```