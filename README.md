# [RIO-Bench] Read or Ignore? A Unified Benchmark for Typographic-Attack Robustness and Text Recognition in Vision-Language Models

Real-world VLMs must decide when to read text and when to ignore it—e.g., reading traffic signs but not being fooled by text-based attacks on objects.
To evaluate this ability, we introduce:

- RIO-VQA: A task that formalizes selective text use under clean and typographic-attack scenarios.

- RIO-Bench: A benchmark providing same-scene counterfactuals (read / ignore variants) by modifying only textual content and question type.

### RIO-VQA: VLMs Must Adaptively Read or Ignore Texts
<img src="./figures/teaser_light.png" alt="RIO-VQA Overview" width="600"/>


### RIO-VQA Taxonomy
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

# RIO-Bench Construction Reproduction / Customization
Details are provided in: ```data_construction/README.md```


