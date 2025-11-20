
import sys
import os
import json
import time
import csv
import re
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.semantic_parser import parse_with_llm
from utils.json_retriever import retrieve_info
from models.llm_with_json import llm_with_json_answer
from models.llm_only import llm_only_answer

QUESTIONS_CSV = Path("evaluation/questions.csv")
EXPECTED_JSON = Path("evaluation/expected.json")
OUTPUT_CSV = Path("evaluation/evaluation_results.csv")
SUMMARY_JSON = Path("evaluation/summary.json")


# ----------------------------- LOADING HELPERS ------------------------------

def load_questions(path: Path) -> List[str]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            q = r.get("question") or r.get("Question")
            if q:
                rows.append(q.strip())
    return rows


def load_expected(path: Path) -> Dict[str, List[Dict]]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_value(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip().lower()


def normalize_obj(obj: Dict[str, Any]) -> Dict[str, str]:
    return {k.lower(): normalize_value(v) for k, v in obj.items()}


def list_of_norm_strings_from_objs(objs: List[Dict[str, Any]]) -> List[str]:
    vals = []
    for o in objs:
        if isinstance(o, dict):
            no = normalize_obj(o)
            for v in no.values():
                vals.append(v)
    return list(dict.fromkeys(vals))


# ------------------------ MATCH EXPECTED RETRIEVAL -------------------------

def retrieved_matches_expected(expected_list: List[Dict], retrieved) -> bool:

    # Normalize to list of dicts
    if retrieved is None:
        retrieved_list = []
    elif isinstance(retrieved, dict):
        retrieved_list = [retrieved]
    elif isinstance(retrieved, list):
        retrieved_list = [x for x in retrieved if isinstance(x, dict)]
    else:
        retrieved_list = []

    if not expected_list and not retrieved_list:
        return True
    if not expected_list and retrieved_list:
        return False
    if expected_list and not retrieved_list:
        return False

    norm_expected = [normalize_obj(e) for e in expected_list]
    norm_retrieved = [normalize_obj(r) for r in retrieved_list]

    for exp in norm_expected:
        matched = False
        for ret in norm_retrieved:
            ok = True
            for k, v in exp.items():
                if v and v not in ret.get(k, ""):
                    ok = False
                    break
            if ok:
                matched = True
                break
        if not matched:
            return False

    return True


# -------------------- HALLUCINATION DETECTOR (PATCHED) ----------------------

def extract_factual_tokens(answer: str) -> List[str]:
    text = (answer or "").lower()
    tokens = set()

    for m in re.findall(r"\$\s*\d+(\.\d+)?", text):
        tokens.add(m)

    for m in re.finditer(r"\d+(\.\d+)?\s*(tb|gb)", text):
        tokens.add(m.group(0).strip())

    for m in re.finditer(r"\b\d+(\.\d+)?\b", text):
        tokens.add(m.group(0))

    for w in re.findall(r"\b[a-z0-9\-_]{3,}\b", text):
        tokens.add(w)

    return [t.strip() for t in tokens]


def detect_hallucination_from_answer(retrieved_objs, answer: str) -> bool:

    # 🛑 SAFETY PATCH
    if not isinstance(retrieved_objs, list):
        retrieved_objs = []
    else:
        retrieved_objs = [r for r in retrieved_objs if isinstance(r, dict)]

    answer = answer or ""
    retrieved_tokens = set(list_of_norm_strings_from_objs(retrieved_objs))

    if not retrieved_objs:
        low = answer.lower()
        if "no matching" in low or "not found" in low or low.strip() == "":
            return False
        return len(extract_factual_tokens(answer)) > 0

    factual_tokens = extract_factual_tokens(answer)
    if not factual_tokens:
        return False

    for t in factual_tokens:
        if t in {"plan", "monthly", "price"}:
            continue
        matched = any(t in rv for rv in retrieved_tokens)
        if not matched:
            return True

    return False


# ---------------------- COMPLETENESS SCORE (PATCHED) -----------------------

def completeness_fraction(retrieved_objs, answer: str) -> float:

    # 🛑 SAFETY PATCH
    if not isinstance(retrieved_objs, list):
        retrieved_objs = []
    else:
        retrieved_objs = [o for o in retrieved_objs if isinstance(o, dict)]

    if not retrieved_objs:
        return 1.0

    ans = (answer or "").lower()
    total = 0
    matched = 0

    for obj in retrieved_objs:
        for k, v in obj.items():
            total += 1
            if str(v).lower() in ans:
                matched += 1

    return round(matched / total, 2) if total else 1.0


# ----------------------------- SAFE WRAPPERS -------------------------------

def safe_parse(q):
    try:
        t0 = time.time()
        out = parse_with_llm(q)
        return out, time.time() - t0, None
    except Exception as e:
        return None, 0, e


def safe_retrieve(parsed):
    try:
        t0 = time.time()
        out = retrieve_info(parsed)
        return out, time.time() - t0, None
    except Exception as e:
        return None, 0, e


def safe_llm_only(q):
    try:
        t0 = time.time()
        out = llm_only_answer(q)
        return out, time.time() - t0, None
    except Exception as e:
        return None, 0, e


def safe_rag(q):
    try:
        t0 = time.time()
        out = llm_with_json_answer(q)
        return out, time.time() - t0, None
    except Exception as e:
        return None, 0, e


# ------------------------------ MAIN LOGIC --------------------------------

def main():

    questions = load_questions(QUESTIONS_CSV)
    expected_map = load_expected(EXPECTED_JSON)

    results = []

    stats = {
        "total": len(questions),
        "correct_retrieval": 0,
        "retrieval_success": 0,
        "parser_consistency": 0,
        "retrieval_consistency": 0,
        "hallucination_llm_only": 0,
        "hallucination_rag": 0,
        "completeness_llm_only": 0,
        "completeness_rag": 0,
    }

    parse_lat = []
    retr_lat = []
    gen_llm_lat = []
    gen_rag_lat = []

    for q in questions:
        print("\n=== Question ===")
        print(q)

        expected = expected_map.get(q, [])

        # Run parser & retriever 3 times to check consistency
        parses = []
        retrieves = []
        p_times = []
        r_times = []

        for _ in range(3):
            p, pt, _ = safe_parse(q)
            parses.append(json.dumps(p, sort_keys=True) if p else None)
            p_times.append(pt)

            if p is not None:
                r, rt, _ = safe_retrieve(p)
            else:
                r, rt = None, 0

            if isinstance(r, dict):
                retrieves.append([r])
            elif isinstance(r, list):
                retrieves.append([x for x in r if isinstance(x, dict)])
            else:
                retrieves.append([])
            r_times.append(rt)

        # Consistency checks
        if parses[0] == parses[1] == parses[2]:
            stats["parser_consistency"] += 1

        if retrieves[0] == retrieves[1] == retrieves[2]:
            stats["retrieval_consistency"] += 1

        rep_retrieved = retrieves[0] if retrieves[0] else []

        if rep_retrieved:
            stats["retrieval_success"] += 1

        if retrieved_matches_expected(expected, rep_retrieved):
            stats["correct_retrieval"] += 1

        # LLM-only and RAG answers
        llm_ans, llm_lat, _ = safe_llm_only(q)
        rag_ans, rag_lat, _ = safe_rag(q)

        gen_llm_lat.append(llm_lat)
        gen_rag_lat.append(rag_lat)
        parse_lat.extend(p_times)
        retr_lat.extend(r_times)

        # Hallucination
        if detect_hallucination_from_answer(rep_retrieved, llm_ans or ""):
            stats["hallucination_llm_only"] += 1
        if detect_hallucination_from_answer(rep_retrieved, rag_ans or ""):
            stats["hallucination_rag"] += 1

        # Completeness
        stats["completeness_llm_only"] += completeness_fraction(rep_retrieved, llm_ans or "")
        stats["completeness_rag"] += completeness_fraction(rep_retrieved, rag_ans or "")

        results.append({
            "question": q,
            "expected": expected,
            "retrieved": rep_retrieved,
            "llm_only": llm_ans,
            "rag": rag_ans
        })

    summary = {
        "total_questions": stats["total"],
        "retrieval_accuracy_percent": round(100 * stats["correct_retrieval"] / stats["total"], 2),
        "retrieval_success_percent": round(100 * stats["retrieval_success"] / stats["total"], 2),
        "parser_consistency_percent": round(100 * stats["parser_consistency"] / stats["total"], 2),
        "retrieval_consistency_percent": round(100 * stats["retrieval_consistency"] / stats["total"], 2),
        "hallucination_rate_llm_only_percent": round(100 * stats["hallucination_llm_only"] / stats["total"], 2),
        "hallucination_rate_rag_percent": round(100 * stats["hallucination_rag"] / stats["total"], 2),
        "avg_completeness_llm_only": round(stats["completeness_llm_only"] / stats["total"], 3),
        "avg_completeness_rag": round(stats["completeness_rag"] / stats["total"], 3),
        "avg_parse_latency": round(sum(parse_lat) / len(parse_lat), 4),
        "avg_retrieve_latency": round(sum(retr_lat) / len(retr_lat), 4),
        "avg_llm_generate_latency": round(sum(gen_llm_lat) / len(gen_llm_lat), 4),
        "avg_rag_generate_latency": round(sum(gen_rag_lat) / len(gen_rag_lat), 4)
    }

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== FINAL SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"Saved: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
