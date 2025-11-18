# evaluation/evaluate_accuracy.py

import sys
import os
import csv
import time
import json
from pathlib import Path

# Add project root to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# Import pipeline functions
try:
    from models.llm_only import llm_only_answer
except Exception:
    from llm_only import llm_only_answer

try:
    from models.llm_with_json import llm_with_json_answer
except Exception:
    from llm_with_json import llm_with_json_answer


# -------------------------------------------------------------------
# CONFIG PATHS
# -------------------------------------------------------------------
QUESTIONS_PATH = Path("evaluation/questions.csv")
OUTPUT_PATH = Path("evaluation/evaluation_results.csv")
SUMMARY_PATH = Path("evaluation/summary.json")


# -------------------------------------------------------------------
# UTILITY FUNCTIONS
# -------------------------------------------------------------------

def load_questions(path: Path):
    """Load evaluation questions & expected keywords."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            q = (r.get("question") or "").strip()
            e = (r.get("expected_keyword") or "").strip()
            if q:
                rows.append({"question": q, "expected": e})
    return rows


def safe_call(func, question, max_retries=2, pause=1):
    """Handles API errors + retry logic."""
    for attempt in range(max_retries + 1):
        try:
            return func(question)
        except Exception as e:
            print(f"[WARN] Error calling {func.__name__}: {e}")
            if attempt < max_retries:
                time.sleep(pause)
            else:
                return f"ERROR: {e}"


def is_correct(expected_keyword, answer):
    """Simple keyword correctness metric."""
    if not expected_keyword:
        return False
    return expected_keyword.lower() in (answer or "").lower()


def detect_hallucination(answer, expected_keyword):
    """
    A simple rule-based hallucination detector:
    - If answer contains something not in JSON retrieval
    - If expected keyword missing
    - If answer says something impossible (like free 2TB)
    """
    if "ERROR" in answer:
        return True
    if "No matching data found" in answer:
        return True
    if expected_keyword.lower() not in answer.lower():
        return True
    return False


def measure_consistency(func, question):
    """Run 3 times and check stability."""
    answers = [
        safe_call(func, question),
        safe_call(func, question),
        safe_call(func, question)
    ]
    return answers, int(answers[0] == answers[1] == answers[2])


def measure_latency(func, question):
    start = time.time()
    ans = safe_call(func, question)
    end = time.time()
    return ans, end - start


def extract_json_object(text):
    """Helper: detect JSON part in RAG output to measure completeness."""
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except:
        return None


def completeness_score(answer):
    """
    Reward answers that mention:
    - Plan name
    - Storage
    - Price
    """
    score = 0
    if any(x in answer.lower() for x in ["tb", "gb", "storage"]):
        score += 1
    if any(x in answer.lower() for x in ["$", "price", "monthly", "annual"]):
        score += 1
    if any(x in answer.lower() for x in ["plan", "plus", "advanced", "premium", "standard"]):
        score += 1
    return score


# -------------------------------------------------------------------
# MAIN EVALUATION LOGIC
# -------------------------------------------------------------------

def main():
    print("\n=== Running Full Evaluation ===")

    questions = load_questions(QUESTIONS_PATH)
    total = len(questions)

    results = []

    # metric counters
    acc_llm_only = acc_rag = 0
    halluc_llm_only = halluc_rag = 0
    cons_llm_only = cons_rag = 0
    comp_llm_only = comp_rag = 0
    latencies_llm_only = []
    latencies_rag = []
    retrieval_success = 0

    for idx, item in enumerate(questions, 1):
        q = item["question"]
        expected = item["expected"]

        print("\n" + "=" * 70)
        print(f"[{idx}/{total}] {q}")
        print(f"Expected keyword → {expected}")

        # ---------------- LLM Only -----------------
        answers_o, consistency_o = measure_consistency(llm_only_answer, q)
        latency_o = []
        for _ in range(3):
            _, lt = measure_latency(llm_only_answer, q)
            latency_o.append(lt)
        llm_ans = answers_o[0]
        correct_o = is_correct(expected, llm_ans)
        halluc_o = detect_hallucination(llm_ans, expected)
        comp_o = completeness_score(llm_ans)

        # accumulate metrics
        acc_llm_only += int(correct_o)
        halluc_llm_only += int(halluc_o)
        cons_llm_only += int(consistency_o)
        comp_llm_only += comp_o
        latencies_llm_only.extend(latency_o)

        # ---------------- RAG -----------------
        answers_r, consistency_r = measure_consistency(llm_with_json_answer, q)
        latency_r = []
        for _ in range(3):
            _, lt = measure_latency(llm_with_json_answer, q)
            latency_r.append(lt)
        rag_ans = answers_r[0]
        correct_r = is_correct(expected, rag_ans)
        halluc_r = detect_hallucination(rag_ans, expected)
        comp_r = completeness_score(rag_ans)

        # detect retrieval success
        if "No matching data" not in rag_ans:
            retrieval_success += 1

        acc_rag += int(correct_r)
        halluc_rag += int(halluc_r)
        cons_rag += int(consistency_r)
        comp_rag += comp_r
        latencies_rag.extend(latency_r)

        # Save per-question result
        results.append({
            "question": q,
            "expected_keyword": expected,
            "llm_only_answer": llm_ans,
            "rag_answer": rag_ans,
            "llm_only_correct": correct_o,
            "rag_correct": correct_r,
            "llm_only_hallucination": halluc_o,
            "rag_hallucination": halluc_r,
            "llm_only_consistency": consistency_o,
            "rag_consistency": consistency_r,
            "llm_only_completeness": comp_o,
            "rag_completeness": comp_r,
            "llm_only_latency_avg": sum(latency_o) / len(latency_o),
            "rag_latency_avg": sum(latency_r) / len(latency_r)
        })

    # ---------------- Final summary ----------------------
    summary = {
        "total_questions": total,

        "accuracy_llm_only": round(100 * acc_llm_only / total, 2),
        "accuracy_rag": round(100 * acc_rag / total, 2),

        "hallucination_llm_only": round(100 * halluc_llm_only / total, 2),
        "hallucination_rag": round(100 * halluc_rag / total, 2),

        "consistency_llm_only": round(100 * cons_llm_only / total, 2),
        "consistency_rag": round(100 * cons_rag / total, 2),

        "latency_mean_llm_only": round(sum(latencies_llm_only) / len(latencies_llm_only), 3),
        "latency_mean_rag": round(sum(latencies_rag) / len(latencies_rag), 3),

        "retrieval_success_rate": round(100 * retrieval_success / total, 2),

        "completeness_llm_only": round(comp_llm_only / total, 2),
        "completeness_rag": round(comp_rag / total, 2)
    }

    # Write CSV file
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # Write summary
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== FINAL SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\nSaved evaluation results & summary.")


if __name__ == "__main__":
    main()
