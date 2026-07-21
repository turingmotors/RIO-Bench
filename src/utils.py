import os
import torch
import random
import numpy as np
from transformers import set_seed
from packaging.version import Version
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode


class Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()

    def isatty(self):
        return getattr(self._streams[0], "isatty", lambda: False)()


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def is_internvl_model(model_name: str) -> bool:
    return "internvl" in (model_name or "").lower()


def is_molmo_model(model_name: str) -> bool:
    return "molmo" in (model_name or "").lower()


def build_internvl_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess_internvl(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_internvl_image(image, input_size=448, max_num=12):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image type for InternVL: {type(image)!r}")

    transform = build_internvl_transform(input_size=input_size)
    images = dynamic_preprocess_internvl(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


def _ensure_pad_token(processor_or_tokenizer):
    tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id


def load_model_and_processor(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map=None):

    model_name_lower = model_name.lower()
    kwargs = dict(torch_dtype=torch_dtype, low_cpu_mem_usage=low_cpu_mem_usage)
    if device_map is not None:
        kwargs["device_map"] = device_map

    # Pass HF token if set (needed for gated/private models).
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    if hf_token:
        kwargs["token"] = hf_token
    proc_kwargs = {"token": hf_token} if hf_token else {}

    if is_internvl_model(model_name):
        import transformers
        from transformers import AutoModel, AutoTokenizer

        if Version(transformers.__version__) < Version("4.52.1"):
            raise RuntimeError(
                f"InternVL requires transformers>=4.52.1, but found {transformers.__version__}."
            )

        processor = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
            **proc_kwargs,
        )
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            **kwargs,
        )

    elif "qwen2_5" in model_name_lower or "qwen2.5" in model_name_lower:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_name, **proc_kwargs)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **kwargs)

    elif "qwen3" in model_name_lower:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_name, **proc_kwargs)
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **kwargs)

    elif "smolvlm" in model_name_lower:
        from transformers import AutoProcessor, AutoModelForVision2Seq

        processor = AutoProcessor.from_pretrained(model_name, **proc_kwargs)
        model = AutoModelForVision2Seq.from_pretrained(model_name, **kwargs)

    elif "minicpm-v" in model_name_lower:
        from transformers import AutoModel, AutoTokenizer

        processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, **proc_kwargs)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, **kwargs)

    elif "llava" in model_name_lower:
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_name, **proc_kwargs)
        model = LlavaForConditionalGeneration.from_pretrained(model_name, **kwargs)

    elif "qwen2" in model_name_lower:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_name, **proc_kwargs)
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, **kwargs)

    elif "llama-3.2" in model_name_lower and "vision" in model_name_lower:
        from transformers import MllamaForConditionalGeneration, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_name, **proc_kwargs)
        model = MllamaForConditionalGeneration.from_pretrained(model_name, **kwargs)

    elif "molmo" in model_name_lower:
        from transformers import AutoModelForCausalLM, AutoProcessor

        # Molmo requires bfloat16; override dtype if float16 was requested.
        molmo_kwargs = {**kwargs, "torch_dtype": torch.bfloat16}
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, **proc_kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **molmo_kwargs)

    elif "gemma-3" in model_name_lower or "gemma3" in model_name_lower:
        from transformers import AutoProcessor, AutoModelForImageTextToText

        # Gemma 3 vision models prefer bfloat16.
        gemma_kwargs = {**kwargs, "torch_dtype": torch.bfloat16}
        processor = AutoProcessor.from_pretrained(model_name, **proc_kwargs)
        model = AutoModelForImageTextToText.from_pretrained(model_name, **gemma_kwargs)

    elif "gemma" in model_name_lower:
        from transformers import AutoProcessor, AutoModelForCausalLM

        processor = AutoProcessor.from_pretrained(model_name, **proc_kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    _ensure_pad_token(processor)

    return model, processor


def set_random_seed(seed):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
