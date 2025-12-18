import os
import argparse
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from PIL import Image as PILImage


def patch_getexif():
    """Monkey-patch PIL to avoid UnicodeDecodeError on getexif."""
    orig = PILImage.Image.getexif

    def safe_getexif(self):
        try:
            return orig(self)
        except UnicodeDecodeError:
            return {}

    PILImage.Image.getexif = safe_getexif


def generate_messages(example, dataset_type, model_name, text_only=False, peft_ver=False):
    """
    Generate one or more message dicts for a single example based on dataset and model.
    Returns a list of {'messages': [...] } entries.
    """
    if "llava" in model_name.lower():
        eos = "</s>"
    else:
        eos = ""

    # Common fields
    context = example.get("hint", "")
    question = example["question"]
    choices = example.get("choices", [])
    options = [chr(ord("A") + i) for i in range(len(choices))]
    choices_str = "\n".join(f"{opt}. {text}" for opt, text in zip(options, choices))

    # Build query string
    if "scienceqa" in dataset_type.lower():
        if "qwen" in model_name.lower():
            ctx = context if context else "N/A"
            query = f"Context: {ctx}\nQuestion: {question}\nOptions: {choices_str}\nAnswer:"
        else:
            post = "\nAnswer with the option's letter from the given choices directly."
            query = f"{context}{question}\n{choices_str}{post}"
    elif "ok-vqa" in dataset_type.lower():
        post = "\nWhen the provided information is insufficient, respond with 'Unanswerable'.\nAnswer the question using a single word or phrase."
        query = f"{question}{post}"
    elif "vizwiz" in dataset_type.lower():
        post = "\nWhen the provided information is insufficient, respond with 'Unanswerable'.\nAnswer the question using a single word or phrase."
        query = f"{question.capitalize()}{post}"
    elif "textvqa" in dataset_type.lower():
        if "qwen" in model_name.lower():
            post = " Answer:"
        else:
            post = "\nAnswer the question with a single word."
        query = f"{question}{post}"
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    # Content elements
    if text_only:
        image_comp = {"type": "text", "text": example.get("image_caption", "") + " "}
    else:
        image_comp = {"type": "image", "image": example.get("image")}
    text_comp = {"type": "text", "text": query}
    user_content = [image_comp, text_comp]

    # Assemble messages
    messages_list = []
    if "scienceqa" in dataset_type.lower():
        # Single response
        answer = example.get("answer", 0)
        resp = options[answer]
        resp_text = resp + eos
        msg = {"messages": [{"role": "user", "content": user_content}, {"role": "assistant", "content": [{"type": "text", "text": resp_text}]}]}
        if peft_ver:
            msg["images"] = [example.get("image")]
        messages_list.append(msg)
    elif "textvqa" in dataset_type.lower():
        # Single response
        answer = example.get("answers", [""])[0]
        resp_text = answer + eos
        msg = {"messages": [{"role": "user", "content": user_content}, {"role": "assistant", "content": [{"type": "text", "text": resp_text}]}]}
        if peft_ver:
            msg["images"] = [example.get("image")]
        messages_list.append(msg)
    else:
        # 複数のanswersから1つだけ選ぶ
        answer = example.get("answers", [""])[0]
        resp_text = (answer if isinstance(answer, str) else options[answer]) + eos
        msg = {"messages": [{"role": "user", "content": user_content}, {"role": "assistant", "content": [{"type": "text", "text": resp_text}]}]}
        if peft_ver:
            msg["images"] = [example.get("image")]
        messages_list.append(msg)
    return messages_list


def generate_messages_instruct(example, model_name, peft_ver=False):
    """
    Generate one or more message dicts for a single example based on dataset and model.
    Returns a list of {'messages': [...] } entries.
    """
    if "llava" in model_name.lower():
        eos = "</s>"
    else:
        eos = ""

    image = example.get("image")
    conversations = example.get("conversations")
    if not image or not conversations:
        raise ValueError("Example must contain 'image' and 'conversations' fields.")

    messages = []
    for i, conv in enumerate(conversations):
        user, value = conv.get("from"), conv.get("value", "")
        if user not in ["human", "gpt"]:
            raise ValueError(f"Unsupported conversation role: {user}")

        if conv["from"] == "human":
            user_content = []
            # 最初のhuman発話には画像を追加
            if i == 0:
                user_content.append({"type": "image", "image": image})
            user_content.append({"type": "text", "text": conv["value"].replace("<image>", "").strip()})
            messages.append({"role": "user", "content": user_content})
        elif conv["from"] == "gpt":
            assistant_content = [{"type": "text", "text": conv["value"] + eos}]
            messages.append({"role": "assistant", "content": assistant_content})
    if peft_ver:
        messages_list = [{"images": [image], "messages": messages}]
    else:
        messages_list = [{"messages": messages}]
    return messages_list


def preprocess_dataset(dataset, dataset_id, model_name, text_only=False, peft_ver=False):
    """
    Generic preprocessing for supported datasets and models.
    """
    patch_getexif()
    ds_lower = dataset_id.lower()
    if "scienceqa" in ds_lower:
        ds_type = "scienceqa"
    elif "ok-vqa" in ds_lower:
        ds_type = "ok-vqa"
    elif "vizwiz" in ds_lower:
        ds_type = "vizwiz"
    elif "textvqa" in ds_lower:
        ds_type = "textvqa"
    elif "typodefense" in ds_lower or "llava_instruct" in ds_lower:
        ds_type = "llava_instruct"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_id}")

    processed = []
    for ex in tqdm(dataset, desc=f"Processing {dataset_id}", total=len(dataset)):
        if ds_type == "llava_instruct":
            msgs = generate_messages_instruct(ex, model_name, peft_ver=peft_ver)
        else:
            msgs = generate_messages(ex, ds_type, model_name, text_only, peft_ver=peft_ver)
        processed.extend(msgs)
    return processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, default="datasets/ScienceQAimg_train", help="🤗 Datasets ID or local path")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to process")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct", help="Vision-language model identifier")
    parser.add_argument("--text_only", action="store_true", help="Use text-only preprocessing (use captions instead of images)")
    args = parser.parse_args()

    # Load dataset
    if os.path.isdir(args.dataset_id):
        dataset = load_from_disk(args.dataset_id)
    else:
        dataset = load_dataset(args.dataset_id, split=args.split, trust_remote_code=True)

    dataset = dataset.select(range(100))

    print(f"Loaded dataset {args.dataset_id} with {len(dataset)} examples.")
    print(f"Processing with model: {args.model_name} (text_only={args.text_only})")
    # Preprocess dataset
    processed_dataset = preprocess_dataset(dataset, args.dataset_id, args.model_name, args.text_only)

    from utils import load_model_and_processor

    _, processor = load_model_and_processor(args.model_name)

    print("message")
    print(processed_dataset[0]["messages"])

    chat = processor.apply_chat_template(processed_dataset[0]["messages"], add_generation_prompt=True, tokenize=False)

    print("Example processed message:")
    print(chat)

    print(f"PAD token ID: {processor.tokenizer.pad_token_id}")
    print(f"EOS token ID: {processor.tokenizer.eos_token_id}")

    eos = processor.batch_decode([processor.tokenizer.eos_token_id], skip_special_tokens=False)[0]
    print(f"EOS token: {eos}")
    eos = processor.batch_decode([processor.tokenizer.pad_token_id], skip_special_tokens=False)[0]
    print(f"PAD token: {eos}")

    # Example: limit for debug
    # dataset_ids = [
    #     "datasets/VizWiz_train_50words",
    #     "datasets/VizWiz_train_200words",
    #     "datasets/ScienceQAimg_train_50words",
    #     "datasets/ScienceQAimg_train_200words",
    #     "datasets/OK-VQA_train_50words",
    #     "datasets/OK-VQA_train_200words",
    # ]

    # for dataset_id in dataset_ids:
    #     dataset = load_from_disk(dataset_id)
    #     print(f"######### Processing dataset: {dataset_id} #########")

    #     print(dataset[1]["image_caption"])

    #     print("\n\n\n")
