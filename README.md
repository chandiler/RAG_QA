# RAG_QA – LLM-Only vs LLM + RAG JSON Question Answering

## 🧩 Project Overview
**RAG_QA** demonstrates how retrieval-augmented generation (RAG) improves the factual reliability of large language models (LLMs).  
It compares two pipelines:

| Mode | Description |
|------|--------------|
| **LLM-only** | The language model directly answers user questions using its internal knowledge — often producing hallucinations. |
| **LLM + RAG (JSON)** | The language model first interprets the user’s intent, Python retrieves factual data from a structured JSON dataset, and the LLM then generates a grounded answer based on that context. |

Dataset: `data/cloud_storage.json`, containing real pricing and feature data for Google Drive, Dropbox, OneDrive, and Box.

---

## ⚙️ Architecture

```
User Question
      │
      ▼
[1] Semantic Parsing (LLM)
 → {"Platform": "Dropbox", "Query": "cheapest"}
      │
      ▼
[2] JSON Retrieval (Python)
 → find factual plans matching conditions
      │
      ▼
[3] Answer Generation (LLM)
 → "The cheapest Dropbox plan is Plus, $15.99/month."
```

---

## 📂 Project Structure
```
RAG_QA/
│
├── data/
│   └── cloud_storage.json
│
├── utils/
│   ├── api_client.py          # OpenAI client & .env loader
│   ├── semantic_parser.py     # LLM-based semantic parsing
│   └── json_retriever.py      # Retrieve plans from JSON
│
├── models/
│   ├── llm_only.py            # Pure LLM answers
│   └── llm_with_json.py       # RAG pipeline (semantic + retrieval + generation)
│
├── main.py                    # CLI entry point
└── .env                       # Contains OPENAI_API_KEY
```

---

## 🛠️ Setup

### 1. Create environment
```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate (Windows)
```

### 2. Install dependencies
```bash
pip install openai python-dotenv
```

### 3. Configure `.env`
```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
```

---

## 🚀 Run Demo
```bash
python main.py
```

Example interaction:
```
Enter your question: Which plan is cheapest on Dropbox?

[LLM-only answer]
Dropbox offers a free basic plan...

[LLM+JSON answer]
[STEP 1] semantic parse → {'Platform': 'Dropbox', 'Query': 'cheapest'}
[STEP 2] retrieved cheapest plan → {'Platform': 'Dropbox', 'PlanName': 'Plus', 'Price': 15.99, 'PlanType': 'Monthly'}
[STEP 3] generated answer:
The cheapest Dropbox plan is Plus, priced at $15.99 per month.
```

---

## 🧠 Key Components

### `semantic_parser.py`
Uses GPT-3.5-Turbo to interpret natural language questions and output structured JSON queries:
```json
{"Platform": "Dropbox", "Query": "cheapest"}
```
Later versions support richer filters such as budget ranges and feature preferences.

### `json_retriever.py`
Executes factual retrieval from `cloud_storage.json` based on parsed conditions  
(e.g., cheapest, budget range, supported features).

### `llm_with_json.py`
Combines semantic parsing + factual retrieval + answer generation — the essence of RAG.

---

## 📊 Expected Behavior
| Scenario | LLM-only | LLM + JSON (RAG) |
|-----------|-----------|------------------|
| “Which plan is cheapest on Dropbox?” | May hallucinate a free 2 GB plan | Correctly answers “Plus $15.99/month” |
| “What’s the largest plan on Google Drive?” | May guess outdated info | Retrieves 2 TB plan from JSON |
| “I want a monthly plan around $20 supporting video uploads.” | Not supported | (Extended parser) extracts budget + features and finds matching plans |

---

## 🧩 Extension Ideas
- Add support for compound filters (budget + features + storage size)  
- Evaluate accuracy vs. hallucination rate on custom question sets  
- Deploy as a cli or web demo (Streamlit / FastAPI)

---

## 📜 License
MIT License © 2025 RAG_QA Team
