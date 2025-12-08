pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

pip install git+https://github.com/unslothai/unsloth.git

pip install git+https://github.com/unslothai/unsloth_zoo.git

pip install \
  numpy \
  pandas \
  matplotlib \
  pillow \
  pyyaml \
  tqdm \
  datasets \
  decord \
  scikit-learn \
  transformers==4.52.4 \
  accelerate==1.10.0 \
  sentencepiece==0.2.1 \
  safetensors==0.6.2 \
  huggingface-hub==0.34.4 \
  hf-transfer==0.1.9 \
  open-clip-torch==3.2.0 \
  lmms-eval==0.4.0 \
  opencv-python-headless==4.11.0.86 \
  pycocotools \
  pycocoevalcap \
  sacrebleu \
  einops \
  regex \
  tiktoken \
  qwen-vl-utils \
  bitsandbytes==0.47.0 \
  xformers==0.0.29.post3 \
  peft==0.17.1 \
  trl==0.21.0 \
  wandb
