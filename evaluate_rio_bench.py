import random
import numpy as np
import argparse
import json
import os

from tqdm import tqdm
import torch
from transformers import set_seed

from utils import disable_torch_init, load_model_and_processor
from PIL import Image

from datasets import load_from_disk

from eval_utils.obj_multiple_choices import evaluate_multiple_choice
from eval_utils.obj_clip_match import clip_match
from eval_utils.text_vqa import evaluate_textvqa
from eval_utils.open_images_classes import open_images_classes


def format_multiturn_prompt(task_type, question, prompt_strategy):
    """
    Convert a single-turn question to a multi-turn conversation format based on the prompt strategy.

    For CoT, we remove suffix in the first turn, and add a suffix in the second turn.
    """
    if prompt_strategy == "1phase-basic":
        # Default prompt strategy
        return [question]
    elif prompt_strategy == "1phase-focus":
            prefix = "Focus on the visual aspects of the image, including colors, shapes, composition, and any notable visual themes. Provide a detailed visual description of the image to answer the following question. Then based on your previous description, please delve deeper into the visual details of the image and include any subtle details or elements that were not covered in your initial description to answer the following question. "
            return [prefix + question]
    elif prompt_strategy == "2phase-focus":
        if task_type == "txt_oe":
            suffix = "Answer the question using a single word or phrase."
        elif task_type == "obj_oe":
            suffix = "Answer only with object names."
        elif task_type == "obj_mc":
            suffix = "Answer with only the option letter (A, B, C, or D)."
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        question_no_suffix = question.replace(suffix, "").strip() # Assumes that `suffix` appears at most once at the end of the question.

        prefix1 = "Focus on the visual aspects of the image, including colors, shapes, composition, and any notable visual themes. Provide a detailed visual description of the image to answer the following question. Then based on your previous description, please delve deeper into the visual details of the image and include any subtle details or elements that were not covered in your initial description to answer the following question. "
        prefix2 = "Solve the problem based on the analysis above. "

        return [prefix1 + question_no_suffix, prefix2 + question]
    else:
        raise ValueError(f"Unknown prompt strategy: {prompt_strategy}")


def get_dataset_for_multiturn(ds, task_type="", prompt_strategy=None):
    """
    Load the dataset for multi-turn conversations.

    1. ds: str, name of the dataset
    2. task_type: str, type of task ("obj_mc", "obj_oe", "txt_oe"
    3. prompt_strategy: str, prompt strategy used in inference
    Returns:
        dataset: dict
            must contain keys: "image", "question"
                1. For "obj_mc" task, must contain keys: "answer", "choices", "question_id", "image_id"
                2. For "obj_oe" task, must contain keys: "answer", "question_id", "image_id"
                3. For "txt_oe" task, must contain keys: "answers", "question_id", "image_id"
    """
    if task_type == "obj_mc":
        images, questions, answers, question_ids, image_ids, choices_list = [], [], [], [], [], []
        for item in tqdm(ds, desc="Loading dataset"):
            images.append(item["image"])
            question = item["question"]
            if prompt_strategy is not None:
                question = format_multiturn_prompt(task_type, question, prompt_strategy)
            else:
                question = [question]
            questions.append(question)
            answers.append(item["answer"])  # correct answer
            question_ids.append(item["question_id"])
            image_ids.append(item["image_id"])
            choices_list.append(item["choices"])  # list of choices
        dataset = {
            "image": images,
            "question": questions,
            "answer": answers,
            "question_id": question_ids,
            "image_id": image_ids,
            "choices": choices_list,
        }
        # print(dataset["question"][:2])
        # print(dataset["answer"][:2])
        # print(dataset["choices"][:2])
        # exit()
    elif task_type == "obj_oe":
        images, questions, answers, question_ids, image_ids, answer2score, attack_word = [], [], [], [], [], [], []
        for item in tqdm(ds, desc="Loading dataset"):
            images.append(item["image"])
            question = item["question"]
            if prompt_strategy is not None:
                question = format_multiturn_prompt(task_type, question, prompt_strategy)
            else:
                question = [question]
            questions.append(question)
            answers.append(item["answers"])  # correct answer
            question_ids.append(item["question_id"])
            image_ids.append(item["image_id"])
            answer2score.append(item["answer2score"])  # dict of {answer: score}
            attack_word.append(item["attack_word"])  # attack word if exists
        dataset = {
            "image": images,
            "question": questions,
            "answers": answers,
            "question_id": question_ids,
            "image_id": image_ids,
            "answer2score": answer2score,
            "attack_word": attack_word,
        }
    elif task_type == "txt_oe":
        images, questions, answers, question_ids, image_ids = [], [], [], [], []
        for item in tqdm(ds, desc="Loading dataset"):
            images.append(item["image"])
            question = item["question"]
            if prompt_strategy is not None:
                question = format_multiturn_prompt(task_type, question, prompt_strategy)
            else:
                question = [question]
            questions.append(question)
            answers.append(item["answers"])  # correct answer
            question_ids.append(item["question_id"])
            image_ids.append(item["image_id"])
        dataset = {
            "image": images,
            "question": questions,
            "answers": answers,
            "question_id": question_ids,
            "image_id": image_ids,
        }

    return dataset


def filter_gt_labels(answer2score, threshold=0.2):
    """
    Filter ground-truth labels based on the answer2score dictionary.
    Only keep labels with score >= threshold, but always keep at least one label.
    """
    answer2score = {k: v for k, v in answer2score.items() if v is not None}
    filtered = [ans for ans, score in answer2score.items() if score >= threshold]
    if len(filtered) == 0:
        max_label = max(answer2score, key=answer2score.get)
        filtered = [max_label]
    return filtered


class RIOBenchEvaluator:
    """
    A class to evaluate vision-language models (VLMs) on RIO-Bench:
    - object multiple choice (obj_mc),
    - object open-ended (obj_oe) via Robust-CLIP-Matc,
    - text open-ended (txt_oe) via TextVQA metrics.
    """
    def __init__(self, task_type, device, is_debug=False):
        assert task_type in ["obj_mc", "obj_oe", "txt_oe"], "Invalid task type. Choose from ['obj_mc', 'obj_oe', 'txt_oe']"
        self.task_type = task_type
        self.device = device
        self.is_debug = is_debug
        self.classes = open_images_classes

    def generate_responses_multiturn(self, target_model, target_processor, img_dataset, question_dataset, max_new_tokens=1024):
        """
        Assume that question_dataset is a list of multi-turn conversations.
            question_dataset: [["hi!", "how are you?"], ["hello!","what is this?"]]
        (If question_dataset is a list of single-turn questions, it will be converted to multi-turn format.)

        Memo: batch size is not supported in this function.
        """
        assert len(img_dataset) == len(question_dataset)

        if isinstance(question_dataset[0], str):
            question_dataset = [[v] for v in question_dataset]  # Convert to multi-turn format

        all_conversations = []
        all_responses = []
        for image, questions in tqdm(zip(img_dataset, question_dataset), desc="Generating responses", total=len(img_dataset)):
            assert image is not None, "Image should not be None for object tasks."
            # Load image
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")

            conversation = []
            responses = []
            for q_idx, q in enumerate(questions):
                # Add user message
                user_msg = {
                    "role": "user",
                    "content": []
                }
                if q_idx == 0 and image is not None:
                    user_msg["content"].append({"type": "image"})
                user_msg["content"].append({"type": "text", "text": q})
                conversation.append(user_msg)

                # Prepare inputs for the model
                prompt = target_processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                inputs = target_processor(
                    images=[image],
                    text=[prompt],
                    return_tensors="pt",
                    truncation=False,
                ).to(self.device)

                # generation kwargs
                eos_id = target_processor.tokenizer.eos_token_id
                gen_kwargs = {
                    "do_sample": True,
                    "temperature": 0.001,
                    "max_new_tokens": max_new_tokens,
                    "eos_token_id": eos_id,
                    "pad_token_id": target_processor.tokenizer.pad_token_id,
                }

                # Generate response
                prompt_len = inputs["input_ids"].shape[-1]
                with torch.no_grad():
                    out_ids = target_model.generate(
                        **inputs,
                        **gen_kwargs,
                        use_cache=True,
                    )

                # Get only the generated tokens
                out_ids = out_ids[:, prompt_len:]
                response = target_processor.decode(out_ids[0], skip_special_tokens=True)

                # Add assistant response to conversation
                conversation.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": response}]
                })
                responses.append(response)


            all_responses.append(responses)
            all_conversations.append(conversation)

            if self.is_debug and len(all_responses) >= 10:
                break

        return all_responses, all_conversations

    def evaluate(self, conversations, responses, data):
        """
        Evaluate binary classification from the generated responses.
        - (a) should always be the correct answer.
        """
        if self.task_type == "obj_mc":
            records = evaluate_multiple_choice(
                conversations, responses, data, allow_text_match=True, 
                gt_key="answer"
            )
        elif self.task_type == "obj_oe":
            examples = []
            for i in range(len(responses)):
                try:
                    gt_labels = filter_gt_labels(data["answer2score"][i], threshold=0.2)
                except Exception as e:
                    print(f"Error in filtering gt_labels for item {i}: {e}")
                    exit()
                examples.append({
                    # "image": item.get("image", None),
                    "question": data["question"][i],
                    "gt_labels": gt_labels,
                    "conversation": conversations[i],
                    "pred_text": responses[i],
                    "image_id": data["image_id"][i],
                    "question_id": data["question_id"][i],
                    "attack_word": data["attack_word"][i],  # may be ""
                })
            records = clip_match(
                self.classes, examples,
                model_name="ViT-B-32", pretrained="openai",
                attack_key="attack_word",
            )
        elif self.task_type == "txt_oe":
            doc = []
            for i in range(len(responses)):
                doc.append({
                    "question_id": data["question_id"][i],
                    "image_id": data["image_id"][i],
                    "question": data["question"][i],
                    "answers": data["answers"][i],  # list of correct answers
                })
            results = [[r] for r in responses]  # each result is a list with a single answer string
            records = evaluate_textvqa(doc, results)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
        return records

    def save_results(self, records, output_dir="./outputs/results/model_name_dummy"):
        os.makedirs(output_dir, exist_ok=True)
        result_path = os.path.join(output_dir, "results.json")
        if self.is_debug:
            result_path = os.path.join(output_dir, "results_debug.json")

        results = {
            "total_samples": len(records["records"]),
            "records": records,
        }
        with open(result_path, "w", encoding="utf-8") as fout:
            json.dump(results, fout, ensure_ascii=False, indent=4)
        print(f"Results saved to {result_path}")

    def run(self, target_model, target_processor, dataset, max_new_tokens=256, output_dir="./outputs/results/model_name_dummy", overwrite=False):
        """
        Run the evaluation process on the given dataset using the target model.

        Args:
            target_model: The vision-language model to evaluate.
            target_processor: The processor for the target model.
            dataset (dict): The dataset containing images and questions, etc.
            batch_size (int): The batch size for evaluation.
            max_new_tokens (int): The maximum number of new tokens to generate.
            output_dir (str): The directory to save the results.
        Returns:
            None
        """
        img_dataset = dataset["image"]
        question_dataset = dataset["question"]

        # Get output
        target_model.eval().to(self.device)
        all_responses, all_conversations = self.generate_responses_multiturn(target_model, target_processor, img_dataset, question_dataset, max_new_tokens)
        torch.cuda.empty_cache()

        # Get the final responses
        responses = [responses_each[-1] for responses_each in all_responses]  # Get the last answer from each conversation

        # Evaluate with guardrail
        records = self.evaluate(all_conversations, responses, dataset)
        torch.cuda.empty_cache()

        # Save results
        self.save_results(records, output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VLMs with guardrail.")
    parser.add_argument("--model_name", type=str, required=True, help="Path to the target VLM model.")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the dataset root directory.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to evaluate.")
    parser.add_argument("--task_type", type=str, default=None, choices=["obj_mc", "obj_oe", "txt_oe"], help="Type of task: obj_mc (object multiple choice), obj_oe (object open-ended), txt_oe (text open-ended).")
    parser.add_argument("--prompt_strategy", type=str, required=True, help="Prompt strategy used in inference.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum new tokens to generate.")
    parser.add_argument("--output_dir", type=str, default="./results/harmless", help="Directory to save results.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results if they exist.")
    parser.add_argument("--is_debug", action="store_true", help="Run in debug mode with a smaller subset of the dataset.")

    return parser.parse_args()



def main():
    import pickle
    disable_torch_init()

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Set output directory
    data_name_clean = args.dataset_name.replace("/", "--")
    OUTPUT_DIR = os.path.join(args.output_dir, data_name_clean, args.prompt_strategy)
    results_path = os.path.join(OUTPUT_DIR, "results.json")
    if os.path.exists(results_path):
        if not args.overwrite:
            print(f"Results already exist at {results_path}. Skipping evaluation.")
            return
        else:
            print(f"Overwriting existing results at {results_path}.")

    # Load target model and processor
    target_model, target_processor = load_model_and_processor(args.model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    print(f"Loaded model: {args.model_name}")
    print(f"Model device: {device}")

    # Load dataset
    dataset_path = os.path.join(args.data_root, args.dataset_name)
    dataset = load_from_disk(dataset_path)
    print(f"Loaded dataset from {dataset_path} with {len(dataset)} samples.")

    # Infer task type if not provided
    if args.task_type is None:
        if "mc_" in args.dataset_name:
             args.task_type = "obj_mc"
        elif "oe_" in args.dataset_name or "open_ended" in args.dataset_name:
            if "obj" in args.dataset_name:
                args.task_type = "obj_oe"
            elif "text" in args.dataset_name or "txt" in args.dataset_name:
                args.task_type = "txt_oe"
        elif "txt_clean" in args.dataset_name or "txt_attack" in args.dataset_name:
            args.task_type = "txt_oe"
        else:
            raise ValueError("Cannot infer task type from dataset. Please specify --task_type.")
    print(f"Task type: {args.task_type}")

    # Prepare dataset for multi-turn conversations
    dataset = get_dataset_for_multiturn(dataset, task_type=args.task_type, prompt_strategy=args.prompt_strategy)

    # Initialize evaluator
    evaluator = RIOBenchEvaluator(args.task_type, device, is_debug=args.is_debug)

    # Run evaluation
    evaluator.run(
        target_model=target_model,
        target_processor=target_processor,
        dataset=dataset,
        max_new_tokens=256,
        output_dir=OUTPUT_DIR,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
