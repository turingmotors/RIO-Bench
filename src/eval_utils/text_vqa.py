# Codes borrowed from:
# https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/tasks/textvqa/utils.py

import datetime
import json
import statistics
from typing import Any, Dict, List

from .vqa_eval_metric import EvalAIAnswerProcessor


def textvqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def _use_normalized_em(doc: Dict[str, Any]) -> bool:
    """
    Decide whether this sample should be evaluated by Normalized EM.
    Enabled for shortcut-diagnosis datasets (e.g., obj_attack_read).
    """
    if str(doc.get("eval_metric", "")).lower() in {"normalized_em", "nem"}:
        return True

    for key in ["attack_level", "subset_name", "dataset_name", "mode", "split_name"]:
        val = str(doc.get(key, "")).lower()
        if any(
            tag in val
            for tag in [
                "obj_attack_read",
                "read_typo_obj_attack",
                "attack_read",
            ]
        ):
            return True
    return False


def _to_answer_list(doc: Dict[str, Any]) -> List[str]:
    answers = doc.get("answers", None)
    if isinstance(answers, list):
        return [str(a) for a in answers if a is not None]
    if "answer" in doc and doc["answer"] is not None:
        return [str(doc["answer"])]
    return []


def _normalized_em_score(doc: Dict[str, Any], pred: str, processor: EvalAIAnswerProcessor) -> float:
    gt_answers = _to_answer_list(doc)
    if not gt_answers:
        return 0.0
    pred_norm = processor(pred)
    gt_norm_set = {processor(a) for a in gt_answers}
    return 1.0 if pred_norm in gt_norm_set else 0.0


def _textvqa_soft_score(doc: Dict[str, Any], pred: str, processor: EvalAIAnswerProcessor) -> float:
    accuracy = 0.0
    if "answers" in doc and doc["answers"] is not None:
        gtAcc = []
        doc_answers = [processor(a) for a in doc["answers"]]
        for i in range(len(doc_answers)):
            otherGTAns = [doc_answers[j] for j in range(len(doc_answers)) if i != j]
            matchingAns = [item for item in otherGTAns if item == pred]
            acc = min(1, float(len(matchingAns)) / 3)
            gtAcc.append(acc)
        accuracy = statistics.mean(gtAcc)
    return accuracy


def textvqa_process_results(doc, result):
    eval_ai_processor = EvalAIAnswerProcessor()
    assert len(result) == 1, f"The result should be a list of length 1, but got {len(result)}."
    resAns = eval_ai_processor(result[0])
    if _use_normalized_em(doc):
        accuracy = _normalized_em_score(doc, result[0], eval_ai_processor)
        metric_type = "normalized_em"
    else:
        accuracy = _textvqa_soft_score(doc, resAns, eval_ai_processor)
        metric_type = "textvqa_soft"

    return {
        "exact_match": accuracy,
        "metric_type": metric_type,
        "submission": {
            "question_id": doc["question_id"],
            "answer": resAns,
        },
    }

def textvqa_aggregate_results(results):
    total = len(results)
    exact_match = sum([result["exact_match"] for result in results])
    accuracy = 100.0 * exact_match / total if total > 0 else 0

    aggregated_results = {
        "total": total,
        "exact_match": exact_match,
        "accuracy": accuracy,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    print(f"TextVQA Evaluation Results: {json.dumps(aggregated_results, indent=2)}")

    return aggregated_results

def evaluate_textvqa(docs, results):
    """
    Evaluate TextVQA results.
    Args:
        docs (list): List of documents, each containing 'question_id', 'image', 'question', and 'answers'.
        results (list): List of results, each being a list with a single answer string.
    Returns:
        dict: Aggregated evaluation results including accuracy, detailed submissions, and all Q&A.
    """
    assert len(docs) == len(results), f"The number of docs ({len(docs)}) should be equal to the number of results ({len(results)})."
    processed_results = []
    submissions = []
    qas = []

    for doc, result in zip(docs, results):
        processed_result = textvqa_process_results(doc, result)
        processed_results.append(processed_result)
        submissions.append(processed_result["submission"])
        qas.append({
            "question_id": doc.get("question_id"),
            "question": doc.get("question"),
            "answers": doc.get("answers"),
            "predicted_answer": result[0] if result else None,
            "exact_match": processed_result["exact_match"],
            "metric_type": processed_result["metric_type"],
        })

    aggregated_results = textvqa_aggregate_results(processed_results)
    aggregated_results["submissions"] = submissions
    aggregated_results["records"] = qas

    return aggregated_results
