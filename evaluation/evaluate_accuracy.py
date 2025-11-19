# evaluation/evaluate_accuracy.py
import sys
import os
import json
import time
import csv
from pathlib import Path

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# Import models
from models.llm_only import llm_only_answer
from models.llm_with_json import llm_with_json_answer

# Paths
QUESTIONS_PATH = Path("evaluation/questions.csv")
EXPECTED_PATH = Path("evaluation/expected.json")
OUTPUT_PATH = Path("evaluation/evaluation_results.csv")
SUMMARY_PATH = Path("evaluation/summary.json")


# ============================================================
# Utility: Load questions
# ============================================================
def load_questions():
    questions = []
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row["question"])
    return questions


# ============================================================
# Load expected JSON results
# ============================================================
def load_expected():
    with open(EXPECTED_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Normalization utilities
# ============================================================
def normalize(obj):
    """Normalize JSON to lowercase string values."""
    if isinstance(obj, dict):
        return {k.lower(): str(v).lower() for k, v in obj.items()}
    return obj


# ============================================================
# Compare expected JSON retrieval with RAG answer
# ============================================================
def retrieval_matches(expected_list, answer_text):
    """Check if expected JSON fields appear in answer."""
    answer_lower = answer_text.lower()

    # If expected empty and RAG answered empty → correct
    if len(expected_list) == 0:
        return "no matching" in answer_lower or answer_lower.strip() == ""

    # Extract expected string fragments
    expected_strings = []
    for obj in expected_list:
        norm = normalize(obj)
        for _, val in norm.items():
            expected_strings.append(val)

    expected_strings = list(set(expected_strings))

    # If ANY expected field appears → count as match
    for val in expected_strings:
        if val in answer_lower:
            return True

    return False


# ============================================================
# Detect hallucination
# ============================================================
def detect_hallucination(expected, rag_answer):
    text = rag_answer.lower()

    # Case 1: expected empty set and model says "nothing found"
    if len(expected) == 0:
        if "no matching" in text or "no plan" in text or "not found" in text:
            return False
        else:
            return True  # hallucination

    # Extract expected fields
    expected_values = []
    for obj in expected:
        for v in obj.values():
            expected_values.append(str(v).lower())

    # Hallucination keywords
    keywords = ["tb", "gb", "$", " plan", "storage", "month"]

    for kw in keywords:
        if kw in text:
            # Check if kw matches ANY expected value
            if not any(kw in ev for ev in expected_values):
                return True

    return False


# ============================================================
# Completeness score
# ============================================================
def completeness_score(expected, rag_answer):
    """Count how many of expected fields appear in answer."""
    if len(expected) == 0:
        return 1

    rag_lower = rag_answer.lower()

    total = 0
    matched = 0

    for obj in expected:
        for _, val in obj.items():
            total += 1
            if str(val).lower() in rag_lower:
                matched += 1

    return round(matched / total, 2) if total > 0 else 1


# ============================================================
# Latency wrapper
# ============================================================
def measure_latency(func, question):
    start = time.time()
    ans = func(question)
    end = time.time()
    return ans, end - start


# ============================================================
# SAFE CALL WRAPPER — prevents crashes
# ============================================================
def safe_rag_call(question):
    """
    Runs RAG safely.
    NEVER crashes evaluation.
    """
    try:
        ans, lat = measure_latency(llm_with_json_answer, question)
        return ans, lat, False  # no error
    except Exception as e:
        print("\n[ERROR] RAG FAILED for question:", question)
        print("Reason:", e)
        return "ERROR_RAG_FAILED", 0, True


# ============================================================
# MAIN EVALUATION LOGIC
# ============================================================
def main():
    questions = load_questions()
    expected_data = load_expected()

    results = []
    metrics = {
        "total": len(questions),
        "accuracy_llm_only": 0,
        "accuracy_rag": 0,
        "hallucination_llm_only": 0,
        "hallucination_rag": 0,
        "consistency_llm_only": 0,
        "consistency_rag": 0,
        "latencies_llm_only": [],
        "latencies_rag": [],
        "retrieval_success": 0,
        "completeness_llm_only": 0,
        "completeness_rag": 0
    }

    for q in questions:
        print("\n=====================================")
        print("QUESTION:", q)

        expected = expected_data.get(q, [])

        # ----------------------------------------------------------
        # LLM ONLY — Safe (no crash risk)
        # ----------------------------------------------------------
        llm_answers = []
        llm_lats = []

        for _ in range(3):
            ans, lat = measure_latency(llm_only_answer, q)
            llm_answers.append(ans)
            llm_lats.append(lat)

        llm_final = llm_answers[0]
        metrics["latencies_llm_only"].extend(llm_lats)

        metrics["accuracy_llm_only"] += int(retrieval_matches(expected, llm_final))
        metrics["hallucination_llm_only"] += int(not retrieval_matches(expected, llm_final))
        metrics["consistency_llm_only"] += int(llm_answers.count(llm_final) == 3)
        metrics["completeness_llm_only"] += completeness_score(expected, llm_final)

        # ----------------------------------------------------------
        # RAG — SAFE WRAPPED
        # ----------------------------------------------------------
        rag_answers = []
        rag_lats = []
        rag_error = False

        for _ in range(3):
            ans, lat, err = safe_rag_call(q)
            rag_answers.append(ans)
            rag_lats.append(lat)
            if err:
                rag_error = True

        rag_final = rag_answers[0]
        metrics["latencies_rag"].extend(rag_lats)

        # RAG correctness
        if not rag_error:
            metrics["retrieval_success"] += 1
            metrics["accuracy_rag"] += int(retrieval_matches(expected, rag_final))
            metrics["hallucination_rag"] += int(detect_hallucination(expected, rag_final))
            metrics["completeness_rag"] += completeness_score(expected, rag_final)
            metrics["consistency_rag"] += int(rag_answers.count(rag_final) == 3)
        else:
            # RAG failed → mark as incorrect but not hallucination
            metrics["accuracy_rag"] += 0
            metrics["hallucination_rag"] += 0
            metrics["completeness_rag"] += 0
            metrics["consistency_rag"] += 0

        # store per-question result
        results.append({
            "question": q,
            "expected": expected,
            "llm_only_answer": llm_final,
            "rag_answer": rag_final
        })


    # ----------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------
    summary = {
        "total_questions": metrics["total"],

        "accuracy_llm_only": round(100 * metrics["accuracy_llm_only"] / metrics["total"], 2),
        "accuracy_rag": round(100 * metrics["accuracy_rag"] / metrics["total"], 2),

        "hallucination_llm_only": round(100 * metrics["hallucination_llm_only"] / metrics["total"], 2),
        "hallucination_rag": round(100 * metrics["hallucination_rag"] / metrics["total"], 2),

        "consistency_llm_only": round(100 * metrics["consistency_llm_only"] / metrics["total"], 2),
        "consistency_rag": round(100 * metrics["consistency_rag"] / metrics["total"], 2),

        "latency_mean_llm_only": round(sum(metrics["latencies_llm_only"]) / len(metrics["latencies_llm_only"]), 3),
        "latency_mean_rag": round(sum(metrics["latencies_rag"]) / len(metrics["latencies_rag"]), 3),

        "retrieval_success_rate": round(100 * metrics["retrieval_success"] / metrics["total"], 2),

        "completeness_llm_only": round(metrics["completeness_llm_only"] / metrics["total"], 2),
        "completeness_rag": round(metrics["completeness_rag"] / metrics["total"], 2)
    }

    # write CSV
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # write summary JSON
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=========== FINAL SUMMARY ===========")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
