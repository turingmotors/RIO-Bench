import os
import torch
import random
import numpy as np
from transformers import set_seed


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


def load_model_and_processor(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True):

    model_name_lower = model_name.lower()

    if "qwen2_5" in model_name_lower or "qwen2.5" in model_name_lower:
        # try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_name)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
        # except:
        #     model_name = os.path.join(model_name, "unsloth")
        #     from unsloth import FastVisionModel

        #     model, processor = FastVisionModel.from_pretrained(
        #         model_name=model_name,
        #         load_in_4bit=True,  # 保存時と一致
        #     )
        #     print(f"Using unsloth model: {model_name}")
    elif "qwen3" in model_name_lower:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_name)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
    elif "smolvlm" in model_name_lower:
        from transformers import AutoProcessor, AutoModelForVision2Seq

        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
    elif "minicpm-v" in model_name_lower:
        from transformers import AutoModel, AutoTokenizer

        processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch_dtype)

    elif "llava" in model_name_lower:
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_name)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

    elif "qwen2" in model_name_lower:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_name)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
    elif "llama-guard" in model_name_lower and "vision" in model_name_lower:
        from transformers import AutoModelForVision2Seq, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
    elif "llama-3.2" in model_name_lower and "vision" in model_name_lower:
        from transformers import MllamaForConditionalGeneration, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_name)
        model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

    elif "gemma" in model_name_lower:
        from transformers import AutoProcessor, AutoModelForCausalLM

        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # Check pad token id
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    return model, processor


def set_random_seed(seed):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
