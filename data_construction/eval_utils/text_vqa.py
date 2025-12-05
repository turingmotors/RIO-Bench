# Codes borrowed from:
# https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/tasks/textvqa/utils.py

import datetime
import json
import statistics

from .vqa_eval_metric import EvalAIAnswerProcessor


def textvqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def textvqa_process_results(doc, result):
    eval_ai_processor = EvalAIAnswerProcessor()
    assert len(result) == 1, f"The result should be a list of length 1, but got {len(result)}."
    resAns = eval_ai_processor(result[0])
    accuracy = 0

    if "answers" in doc and doc["answers"] is not None:
        gtAcc = []

        for i in range(len(doc["answers"])):
            doc["answers"][i] = eval_ai_processor(doc["answers"][i])

        for i in range(len(doc["answers"])):
            otherGTAns = [doc["answers"][j] for j in range(len(doc["answers"])) if i != j]
            matchingAns = [item for item in otherGTAns if item == resAns]
            acc = min(1, float(len(matchingAns)) / 3)
            gtAcc.append(acc)
        accuracy = statistics.mean(gtAcc)

    return {
        "exact_match": accuracy,
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
        })

    aggregated_results = textvqa_aggregate_results(processed_results)
    aggregated_results["submissions"] = submissions
    aggregated_results["records"] = qas

    return aggregated_results