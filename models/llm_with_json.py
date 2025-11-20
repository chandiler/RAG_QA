from utils.semantic_parser import parse_with_llm
from utils.json_retriever import retrieve_info
from utils.api_client import client
import json


def llm_with_json_answer(question):
    print("\n STEP 1: === Semantic Parsing ===")
    parsed = parse_with_llm(question)
    print("[DEBUG] semantic parse →", parsed)

    print("\n STEP 2:=== JSON Retrieval ===")
    retrieved = retrieve_info(parsed)
    print("[DEBUG] retrieved JSON slice →", retrieved)

    if not retrieved:
        return "No matching data found."

    print("\n STEP 3: === Answer Generation ===")
    prompt = (
        f"User question: {question}\n"
        f"Use ONLY the following factual data:\n"
        f"{json.dumps(retrieved, indent=2)}\n"
        "Generate a natural-language answer based on it."
    )
    print("[DEBUG] prompt sent to GPT:\n", prompt)
    print("\n")

    resp = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
    )

    answer = resp.choices[0].message.content.strip()
    print("\n STEP 4: === Final Answer ===")
    return answer
