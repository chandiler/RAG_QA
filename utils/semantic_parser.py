import json
from utils.api_client import client


def parse_with_llm(question):
    """
    让 GPT-3.5 把自然语言问题转成结构化查询条件
    """
    system_prompt = (
        "You are a semantic parser for a cloud storage QA system.\n"
        "Extract platform and query type from the user's question.\n"
        "Platforms: ['Google Drive', 'Dropbox', 'OneDrive', 'Box'].\n"
        "Possible queries: 'cheapest', 'largest', 'free', 'features'.\n"
        'Return ONLY a JSON object like {"Platform": "Dropbox", "Query": "cheapest"}.'
    )

    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )

    raw = resp.choices[0].message.content.strip()
    print(f"[DEBUG] raw semantic parse output: {raw}")

    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {"Platform": None, "Query": None}

    print(f"[STEP 1] semantic parse result → {parsed}")
    return parsed
