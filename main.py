from models.llm_only import llm_only_answer
from models.llm_with_json import llm_with_json_answer


def main():
    print("=== Grounded QA System (LLM-only vs LLM+RAG) ===\n")
    while True:
        q = input("Enter your question (or 'exit'): ")
        if q.lower() == "exit":
            break

        print("\n[LLM-only answer]")
        print(llm_only_answer(q))

        print("\n[LLM+JSON answer]")
        print(llm_with_json_answer(q))

        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
