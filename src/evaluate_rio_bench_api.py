import argparse
import base64
import io
import json
import mimetypes
import os
import random
import signal
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.evaluate_rio_bench import (
    RIOBenchEvaluator,
    _load_dataset_any,
    get_dataset_for_multiturn,
)
from src.utils import disable_torch_init


def _infer_task_type(dataset_name: str) -> str:
    parts = [p for p in dataset_name.split("/") if p]
    base_name = parts[-1] if parts else dataset_name
    group_name = parts[1] if len(parts) >= 2 else ""

    if group_name.startswith("txt_"):
        return "txt_oe"
    if group_name.startswith("obj_"):
        if base_name.startswith("mc_") or "mcq" in base_name:
            return "obj_mc"
        if base_name.startswith("oe_") or "open_ended" in base_name:
            return "obj_oe"
    if "/txt_" in f"/{dataset_name}/" or "open_ended" in dataset_name:
        return "txt_oe"
    if "/obj_" in f"/{dataset_name}/":
        if "mc_" in base_name or "mcq" in base_name:
            return "obj_mc"
        if "oe_" in base_name or "open_ended" in base_name:
            return "obj_oe"
    raise ValueError(f"Cannot infer task type from dataset {dataset_name}. Please specify --task_type.")


def _image_to_bytes_and_mime(image: Any) -> Tuple[bytes, str]:
    if isinstance(image, str):
        mime_type = mimetypes.guess_type(image)[0] or "image/png"
        with open(image, "rb") as fin:
            return fin.read(), mime_type

    if not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image type: {type(image)}")

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue(), "image/png"


def _extract_openai_output_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text.strip()

    chunks: List[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) == "output_text":
                value = getattr(content, "text", None)
                if value:
                    chunks.append(value)
    return "\n".join(chunks).strip()


def _extract_gemini_output_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return text.strip()

    chunks: List[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        if content is None:
            continue
        for part in getattr(content, "parts", []) or []:
            value = getattr(part, "text", None)
            if value:
                chunks.append(value)
    return "\n".join(chunks).strip()


def _get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if hasattr(value, "to_json_dict"):
        return _jsonable(value.to_json_dict())
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return str(value)


def _extract_gemini_response_metadata(response: Any) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    candidates_meta = []
    for candidate in getattr(response, "candidates", []) or []:
        candidates_meta.append(
            {
                "finish_reason": _jsonable(_get_attr_or_key(candidate, "finish_reason")),
                "safety_ratings": _jsonable(_get_attr_or_key(candidate, "safety_ratings")),
                "finish_message": _jsonable(_get_attr_or_key(candidate, "finish_message")),
            }
        )
    if candidates_meta:
        metadata["candidates"] = candidates_meta
    prompt_feedback = getattr(response, "prompt_feedback", None)
    if prompt_feedback is not None:
        metadata["prompt_feedback"] = _jsonable(prompt_feedback)
    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata is not None:
        metadata["usage_metadata"] = _jsonable(usage_metadata)
    return metadata


def _debug_suffix(is_debug: bool) -> str:
    return "_debug" if is_debug else ""


def _load_response_checkpoints(path: str) -> Dict[int, Dict[str, Any]]:
    checkpoints: Dict[int, Dict[str, Any]] = {}
    if not os.path.exists(path):
        return checkpoints
    with open(path, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as err:
                print(f"Skipping invalid checkpoint line {line_no} in {path}: {err}")
                continue
            sample_idx = record.get("sample_idx")
            if isinstance(sample_idx, int):
                checkpoints[sample_idx] = record
    return checkpoints


def _append_response_checkpoint(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()
        os.fsync(fout.fileno())


class BaseVisionAPIClient:
    def __init__(
        self,
        model_name: str,
        max_output_tokens: int,
        temperature: float,
        max_retries: int,
        retry_sleep: float,
        request_timeout: float,
    ):
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.request_timeout = request_timeout

    def _run_with_timeout(self, fn):
        if self.request_timeout <= 0:
            return fn()

        def _handle_timeout(signum, frame):
            raise TimeoutError(f"API request timed out after {self.request_timeout} seconds")

        old_handler = signal.signal(signal.SIGALRM, _handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, self.request_timeout)
        try:
            return fn()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

    def _with_retries(self, fn):
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._run_with_timeout(fn)
            except Exception as err:  # noqa: BLE001
                last_err = err
                if attempt == self.max_retries:
                    break
                print(f"API request failed on attempt {attempt}/{self.max_retries}: {err}")
                time.sleep(self.retry_sleep)
        raise RuntimeError(f"API request failed after {self.max_retries} attempts: {last_err}") from last_err

    def generate_multiturn(self, image: Any, questions: List[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
        raise NotImplementedError


class OpenAIVisionClient(BaseVisionAPIClient):
    def __init__(
        self,
        model_name: str,
        max_output_tokens: int,
        temperature: float,
        max_retries: int,
        retry_sleep: float,
        request_timeout: float,
        api_key: str = "",
    ):
        super().__init__(model_name, max_output_tokens, temperature, max_retries, retry_sleep, request_timeout)
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key or None)

    def generate_multiturn(self, image: Any, questions: List[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
        image_bytes, mime_type = _image_to_bytes_and_mime(image)
        image_data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}"

        previous_response_id: Optional[str] = None
        responses: List[str] = []
        conversation: List[Dict[str, Any]] = []

        for turn_idx, question in enumerate(questions):
            content: List[Dict[str, Any]] = []
            if turn_idx == 0:
                content.append({"type": "input_image", "image_url": image_data_url, "detail": "high"})
            content.append({"type": "input_text", "text": question})

            conversation.append(
                {
                    "role": "user",
                    "content": ([{"type": "image"}] if turn_idx == 0 else []) + [{"type": "text", "text": question}],
                }
            )

            def _call():
                kwargs: Dict[str, Any] = {
                    "model": self.model_name,
                    "input": [{"role": "user", "content": content}],
                    "max_output_tokens": self.max_output_tokens,
                    "temperature": self.temperature,
                }
                if previous_response_id is not None:
                    kwargs["previous_response_id"] = previous_response_id
                return self.client.responses.create(**kwargs)

            response = self._with_retries(_call)
            response_text = _extract_openai_output_text(response)
            previous_response_id = getattr(response, "id", None)

            conversation.append({"role": "assistant", "content": [{"type": "text", "text": response_text}]})
            responses.append(response_text)

        return responses, conversation


class GeminiVisionClient(BaseVisionAPIClient):
    def __init__(
        self,
        model_name: str,
        max_output_tokens: int,
        temperature: float,
        max_retries: int,
        retry_sleep: float,
        request_timeout: float,
        api_key: str = "",
        thinking_budget: Optional[int] = None,
    ):
        super().__init__(model_name, max_output_tokens, temperature, max_retries, retry_sleep, request_timeout)
        self.thinking_budget = thinking_budget
        self.sdk_mode = None
        try:
            from google import genai
            from google.genai import types

            self.sdk_mode = "google_genai"
            self.genai = genai
            self.types = types
            self.client = genai.Client(api_key=api_key or None)
        except ImportError:
            try:
                import google.generativeai as genai

                self.sdk_mode = "google_generativeai"
                self.genai = genai
                self.types = None
                genai.configure(api_key=api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
                self.client = genai.GenerativeModel(model_name)
            except ImportError as err:
                raise ImportError(
                    "Gemini SDK is not installed. Install either `google-genai` or `google-generativeai`."
                ) from err

    def generate_multiturn(self, image: Any, questions: List[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
        history: List[Any] = []
        responses: List[str] = []
        conversation: List[Dict[str, Any]] = []

        if self.sdk_mode == "google_genai":
            image_bytes, mime_type = _image_to_bytes_and_mime(image)

            for turn_idx, question in enumerate(questions):
                parts: List[Any] = []
                if turn_idx == 0:
                    parts.append(self.types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
                parts.append(self.types.Part(text=question))
                history.append(self.types.Content(role="user", parts=parts))

                conversation.append(
                    {
                        "role": "user",
                        "content": ([{"type": "image"}] if turn_idx == 0 else []) + [{"type": "text", "text": question}],
                    }
                )

                def _call():
                    thinking_config = None
                    if self.thinking_budget is not None:
                        thinking_config = self.types.ThinkingConfig(thinking_budget=self.thinking_budget)
                    return self.client.models.generate_content(
                        model=self.model_name,
                        contents=history,
                        config=self.types.GenerateContentConfig(
                            temperature=self.temperature,
                            max_output_tokens=self.max_output_tokens,
                            thinking_config=thinking_config,
                        ),
                    )

                response = self._with_retries(_call)
                response_text = _extract_gemini_output_text(response)
                response_metadata = _extract_gemini_response_metadata(response)

                history.append(self.types.Content(role="model", parts=[self.types.Part(text=response_text)]))
                assistant_msg = {"role": "assistant", "content": [{"type": "text", "text": response_text}]}
                if response_metadata:
                    assistant_msg["metadata"] = response_metadata
                conversation.append(assistant_msg)
                responses.append(response_text)
        else:
            pil_image = Image.open(image).convert("RGB") if isinstance(image, str) else image.convert("RGB")
            chat = self.client.start_chat(history=[])

            for turn_idx, question in enumerate(questions):
                parts: List[Any] = []
                if turn_idx == 0:
                    parts.append(pil_image)
                parts.append(question)

                conversation.append(
                    {
                        "role": "user",
                        "content": ([{"type": "image"}] if turn_idx == 0 else []) + [{"type": "text", "text": question}],
                    }
                )

                def _call():
                    generation_config = {
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_output_tokens,
                    }
                    if self.thinking_budget is not None:
                        generation_config["thinking_config"] = {"thinking_budget": self.thinking_budget}
                    return chat.send_message(parts, generation_config=generation_config)

                response = self._with_retries(_call)
                response_text = _extract_gemini_output_text(response)
                response_metadata = _extract_gemini_response_metadata(response)

                assistant_msg = {"role": "assistant", "content": [{"type": "text", "text": response_text}]}
                if response_metadata:
                    assistant_msg["metadata"] = response_metadata
                conversation.append(assistant_msg)
                responses.append(response_text)

        return responses, conversation


class APIBasedRIOBenchEvaluator(RIOBenchEvaluator):
    def __init__(self, task_type: str, api_client: BaseVisionAPIClient, is_debug: bool = False):
        super().__init__(task_type=task_type, device=torch.device("cpu"), is_debug=is_debug)
        self.api_client = api_client

    def generate_responses_multiturn(self, img_dataset, question_dataset, output_dir: Optional[str] = None, overwrite: bool = False):
        assert len(img_dataset) == len(question_dataset)
        if isinstance(question_dataset[0], str):
            question_dataset = [[v] for v in question_dataset]

        checkpoint_path = None
        checkpoints: Dict[int, Dict[str, Any]] = {}
        if output_dir is not None:
            checkpoint_path = os.path.join(output_dir, f"responses{_debug_suffix(self.is_debug)}.jsonl")
            if overwrite and os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            checkpoints = _load_response_checkpoints(checkpoint_path)
            if checkpoints:
                print(f"Loaded {len(checkpoints)} response checkpoints from {checkpoint_path}")

        eval_limit = min(10, len(img_dataset)) if self.is_debug else len(img_dataset)
        all_responses: List[Optional[List[str]]] = [None] * eval_limit
        all_conversations: List[Optional[List[Dict[str, Any]]]] = [None] * eval_limit

        for sample_idx, (image, questions) in tqdm(
            enumerate(zip(img_dataset[:eval_limit], question_dataset[:eval_limit])),
            desc="Generating API responses",
            total=eval_limit,
        ):
            cached = checkpoints.get(sample_idx)
            if cached is not None:
                all_responses[sample_idx] = cached.get("responses", [])
                all_conversations[sample_idx] = cached.get("conversation", [])
                continue

            responses, conversation = self.api_client.generate_multiturn(image=image, questions=questions)
            all_responses[sample_idx] = responses
            all_conversations[sample_idx] = conversation

            if checkpoint_path is not None:
                _append_response_checkpoint(
                    checkpoint_path,
                    {
                        "sample_idx": sample_idx,
                        "question_id": _jsonable(self._field_at("question_id", sample_idx)),
                        "image_id": _jsonable(self._field_at("image_id", sample_idx)),
                        "responses": responses,
                        "conversation": conversation,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )

        return [r or [] for r in all_responses], [c or [] for c in all_conversations]

    def _field_at(self, field_name: str, idx: int) -> Any:
        data = getattr(self, "_checkpoint_dataset", {})
        values = data.get(field_name)
        if isinstance(values, list) and idx < len(values):
            return values[idx]
        return None

    def run(self, dataset, output_dir: str, overwrite: bool = False, skip_obj_oe_eval: bool = False, filename: str = "results.json"):
        self._checkpoint_dataset = dataset
        all_responses, all_conversations = self.generate_responses_multiturn(
            dataset["image"],
            dataset["question"],
            output_dir=output_dir,
            overwrite=overwrite,
        )
        if skip_obj_oe_eval and self.task_type == "obj_oe":
            print("Skipping obj_oe CLIP evaluation. Responses were saved to checkpoint JSONL.")
            return

        responses = [responses_each[-1] if responses_each else "" for responses_each in all_responses]
        records = self.evaluate(all_conversations, responses, dataset)
        self.save_results(records, output_dir, filename=filename)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate closed VLM APIs on RIO-Bench.")
    parser.add_argument("--api_provider", type=str, required=True, choices=["openai", "gemini"], help="Closed-model API provider.")
    parser.add_argument("--model_name", type=str, required=True, help="API model identifier, e.g. gpt-4.1 or gemini-2.5-pro.")
    parser.add_argument("--api_key", type=str, default="", help="API key. If empty, the provider SDK environment variable is used.")
    parser.add_argument("--data_root", type=str, default="", help="Path to the dataset root directory (optional).")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to evaluate.")
    parser.add_argument("--repo_id", type=str, default="turing-motors/RIO-Bench", help="HF dataset repo id.")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""), help="HF token for private repos.")
    parser.add_argument("--task_type", type=str, default=None, choices=["obj_mc", "obj_oe", "txt_oe"], help="Type of task.")
    parser.add_argument("--prompt_strategy", type=str, required=True, help="Prompt strategy used in inference.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum output tokens per turn.")
    parser.add_argument("--thinking_budget", type=int, default=None, help="Gemini thinking budget. For gemini-2.5-pro, use at least 128.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for the API model.")
    parser.add_argument("--output_dir", type=str, default="./results/api", help="Directory to save results.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum API retry count per request.")
    parser.add_argument("--retry_sleep", type=float, default=3.0, help="Sleep time between retries in seconds.")
    parser.add_argument("--request_timeout", type=float, default=180.0, help="Timeout for one API request in seconds. Set <=0 to disable.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results if they exist.")
    parser.add_argument("--is_debug", action="store_true", help="Run in debug mode with a smaller subset of the dataset.")
    parser.add_argument("--skip_obj_oe_eval", action="store_true", help="For obj_oe tasks, save API responses and skip CLIP evaluation.")
    parser.add_argument("--clip_only", action="store_true", help="Load responses from existing checkpoint and run CLIP evaluation only. No API calls are made; errors if responses are incomplete.")
    parser.add_argument("--question_ids_file", type=str, default="", help="Path to JSON file with list of question_ids to evaluate (subset evaluation).")
    return parser.parse_args()


def build_api_client(args) -> BaseVisionAPIClient:
    if args.api_provider == "openai":
        return OpenAIVisionClient(
            model_name=args.model_name,
            max_output_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
            request_timeout=args.request_timeout,
            api_key=args.api_key,
        )
    if args.api_provider == "gemini":
        return GeminiVisionClient(
            model_name=args.model_name,
            max_output_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
            request_timeout=args.request_timeout,
            api_key=args.api_key,
            thinking_budget=args.thinking_budget,
        )
    raise ValueError(f"Unsupported provider: {args.api_provider}")


def main():
    disable_torch_init()
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_name_clean = args.dataset_name.replace("/", "--")
    model_name_clean = args.model_name.replace("/", "__")
    output_root = os.path.join(args.output_dir, args.api_provider, model_name_clean)
    output_dir = os.path.join(output_root, data_name_clean, args.prompt_strategy)
    results_filename = "results.json"
    results_path = os.path.join(output_dir, results_filename)

    if os.path.exists(results_path) and not args.overwrite:
        print(f"Results already exist at {results_path}. Skipping evaluation.")
        return

    dataset = _load_dataset_any(args)
    print(f"Loaded dataset with {len(dataset)} samples from {args.dataset_name}")

    if args.question_ids_file:
        import json as _json
        with open(args.question_ids_file) as f:
            question_ids_filter = set(int(qid) for qid in _json.load(f))
        dataset = dataset.filter(lambda ex: int(ex["question_id"]) in question_ids_filter)
        print(f"Filtered dataset to {len(dataset)} samples matching {args.question_ids_file}")

    if args.task_type is None:
        args.task_type = _infer_task_type(args.dataset_name)
    print(f"Task type: {args.task_type}")

    dataset = get_dataset_for_multiturn(dataset, task_type=args.task_type, prompt_strategy=args.prompt_strategy)
    dataset["dataset_name"] = [args.dataset_name for _ in range(len(dataset["question"]))]

    # Check if all responses are already cached so we can skip API client init when not needed.
    checkpoint_path = os.path.join(output_dir, f"responses{_debug_suffix(args.is_debug)}.jsonl")
    eval_limit = min(10, len(dataset["question"])) if args.is_debug else len(dataset["question"])
    cached_count = len(_load_response_checkpoints(checkpoint_path)) if os.path.exists(checkpoint_path) and not args.overwrite else 0
    all_cached = cached_count >= eval_limit

    if args.clip_only:
        if not all_cached:
            raise RuntimeError(
                f"--clip_only requires all {eval_limit} responses to be cached, "
                f"but only {cached_count} found in {checkpoint_path}"
            )
        print(f"--clip_only: all {cached_count} responses cached. Skipping API client init.")
        api_client = None
    elif all_cached:
        print(f"All {cached_count} responses already cached; skipping API client init.")
        api_client = None
    else:
        api_client = build_api_client(args)
    evaluator = APIBasedRIOBenchEvaluator(task_type=args.task_type, api_client=api_client, is_debug=args.is_debug)
    evaluator.run(
        dataset=dataset,
        output_dir=output_dir,
        overwrite=args.overwrite,
        skip_obj_oe_eval=args.skip_obj_oe_eval,
        filename=results_filename,
    )

    metadata = {
        "api_provider": args.api_provider,
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "prompt_strategy": args.prompt_strategy,
        "task_type": args.task_type,
        "max_new_tokens": args.max_new_tokens,
        "thinking_budget": args.thinking_budget,
        "request_timeout": args.request_timeout,
        "skip_obj_oe_eval": args.skip_obj_oe_eval,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    metadata_path = os.path.join(output_dir, "run_config.json")
    with open(metadata_path, "w", encoding="utf-8") as fout:
        json.dump(metadata, fout, ensure_ascii=False, indent=2)
    print(f"Saved run metadata to {metadata_path}")


if __name__ == "__main__":
    main()
