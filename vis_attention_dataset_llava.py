from PIL import Image
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn.functional as F
import os
import glob
import numpy as np
from tqdm import tqdm
import json

IMAGE_TOKEN_INDEX = 32000
NUM_IMAGE_TOKENS = 576


def compute_pure_attention(model, processor, image, text):
    device = model.device
    model.eval()

    # Process image for the text
    # inputs = processor(text=text, images=image, return_tensors="pt").to(device)
    inputs, gen_kwargs = format_input(image, text, processor)
    pos = inputs["input_ids"][0].tolist().index(IMAGE_TOKEN_INDEX)
    with torch.no_grad():
        attention = model(**inputs, **gen_kwargs, output_attentions=True)["attentions"]

    # Get attentions of the each layer
    image_attn_list = [att[0, :, -1, pos : pos + NUM_IMAGE_TOKENS].mean(dim=0).to(torch.float32) for att in attention]

    return image_attn_list


def compute_relative_attention(model, processor, image, text, general_text):
    image_attn_list = compute_pure_attention(model, processor, image, text)
    general_image_attn_list = compute_pure_attention(model, processor, image, general_text)

    # Compute relative attention
    relative_image_attn_list = [image_attn / general_image_attn for image_attn, general_image_attn in zip(image_attn_list, general_image_attn_list)]

    return relative_image_attn_list


def compute_gradient_attention(model, processor, image, text):
    device = model.device
    model.eval()

    # Process image for the text
    # inputs = processor(text=text, images=image, return_tensors="pt").to(device)
    inputs, gen_kwargs = format_input(image, text, processor)
    pos = inputs["input_ids"][0].tolist().index(IMAGE_TOKEN_INDEX)

    # Compute loss
    outputs = model(**inputs, **gen_kwargs, output_attentions=True)
    CE = torch.nn.CrossEntropyLoss()
    zero_logit = outputs.logits[:, -1, :]
    true_class = torch.argmax(zero_logit, dim=1)
    loss = -CE(zero_logit, true_class)

    # Compute attention and gradients at each layer
    grad_attn_list = []
    for attention in outputs["attentions"]:
        grads = torch.autograd.grad(loss, attention, retain_graph=True)
        grad_attn = attention * F.relu(grads[0])

        # Compute attention maps
        grad_attn_map = grad_attn[0, :, -1, pos : pos + NUM_IMAGE_TOKENS].mean(dim=0).to(torch.float32)

        grad_attn_list.append(grad_attn_map)

    return grad_attn_list


def visualize_attention(model, processor, image, text, normalize=False, attn_type="pure", general_text=None, out_dir="vis/attention_maps/llava-1.5-7b-hf"):
    # Process image
    if isinstance(image, str):
        image = Image.open(image)

    # Compute attention maps
    if attn_type == "pure":
        image_attn_list = compute_pure_attention(model, processor, image, text)
    elif attn_type == "relative":
        if general_text is None:
            general_text = "Write a general description of the image. Answer the question using a single word or phrase."
        image_attn_list = compute_relative_attention(model, processor, image, text, general_text)
    elif attn_type == "gradient":
        image_attn_list = compute_gradient_attention(model, processor, image, text)
    else:
        raise ValueError(f"Invalid attention type: {attn_type}")

    # Normalize the attention map to the range [0, 1]
    if normalize:
        image_attn_list = [(attn - attn.min()) / (attn.max() - attn.min() + 1e-8) for attn in image_attn_list]

    # Calculate the number of patches and grid size (assumes a square grid)
    num_patches = image_attn_list[0].shape[0]
    grid_size = int(num_patches**0.5)

    # Reshape attention weights into a 2D grid and move to CPU as a numpy array
    attn_map_list = [attn.reshape(grid_size, grid_size).detach().cpu().numpy() for attn in image_attn_list]

    # Overlap the attention map on the image and save the plot at each layer
    for i, attn_map in enumerate(attn_map_list):
        plt.imshow(image)
        plt.imshow(attn_map, alpha=0.75, interpolation="none", extent=[0, image.width, image.height, 0], cmap="viridis")
        plt.axis("off")

        # Save the plot
        save_dir = f"{out_dir}/{attn_type}"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/layer_{i}.jpg", bbox_inches="tight", pad_inches=0)
        plt.close()


def remove_image_extensions(text):
    text = text.replace(".jpg", "")
    text = text.replace(".png", "")
    return text  


def format_input(image, q, processor, max_new_tokens=256, device="cuda"):
    conversation = []

    # Add user message
    user_msg = {
        "role": "user",
        "content": []
    }
    user_msg["content"].append({"type": "image"})
    user_msg["content"].append({"type": "text", "text": q})
    conversation.append(user_msg)

    # Prepare inputs for the model
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        images=[image],
        text=[prompt],
        return_tensors="pt",
        truncation=False,
    ).to(device)

    # generation kwargs
    eos_id = processor.tokenizer.eos_token_id
    gen_kwargs = {
        "do_sample": True,
        "temperature": 0.001,
        "max_new_tokens": max_new_tokens,
        "eos_token_id": eos_id,
        "pad_token_id": processor.tokenizer.pad_token_id,
    }

    return inputs, gen_kwargs



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--normalize", action="store_true")
    # parser.add_argument("--attn_type", type=str, default="pure")
    parser.add_argument("--general_text", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--datasets", type=str, nargs='+', default=[])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model = LlavaForConditionalGeneration.from_pretrained(args.model, attn_implementation="eager").to(device)
    # processor = AutoProcessor.from_pretrained(args.model, revision='a272c74') # Use the specific revision for LLaVA 1.5

    from transformers import AutoProcessor, LlavaForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)

    from datasets import load_from_disk

    qid_to_show = [
        35051, 35110, 35123, 35158, 35290, 35552, 35633, 35860, 35902, 36215
    ]

    # Load the annotation file
    for dataset_name in args.datasets:
        print("Dataset:", dataset_name)

        # Load dataset
        dataset_path = os.path.join(args.data_root, dataset_name)
        dataset = load_from_disk(dataset_path)
        print(f"Loaded dataset from {dataset_path} with {len(dataset)} samples.")

        for i, item in enumerate(tqdm(dataset)):
            image = item["image"]
            question_id = item["question_id"]
            image_id = item["image_id"]
            question = item["question"]

            if question_id not in qid_to_show:
                continue

            image_name = f"q{question_id}_img{image_id}"
            dataset_name = dataset_name.replace("/", "_")

            ###############################################
            # Get output example
            # question = "<image>\nUSER: " + question + "\nASSISTANT:"
            inputs, gen_kwargs = format_input(image, question, processor)
            outputs = model.generate(**inputs, **gen_kwargs)
            generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            print(f"Input text: {question}")
            print(f"Generated text: {generated_text}")

            SAVE_DIR = f"vis/attention_maps/{image_name}/{dataset_name}/{args.model.replace('/', '_')}"
            os.makedirs(SAVE_DIR, exist_ok=True)
            # visualize_attention(model, processor, image, question, args.normalize, args.attn_type, args.general_text, SAVE_DIR)
            # for attn_type in ["pure", "relative", "gradient"]:
            for attn_type in ["relative"]:
                visualize_attention(model, processor, image, question, args.normalize, attn_type, args.general_text, SAVE_DIR)

            # save orig image
            image.save(f"{SAVE_DIR}/original_image.jpg")

            # save model input/output in json
            out_dict = {
                "question": question,
                "generated_text": generated_text
            }
            with open(f"{SAVE_DIR}/model_io.json", "w") as f:
                json.dump(out_dict, f, indent=4)

            # if i == 10:
            #     break


if __name__ == "__main__":
    main()
