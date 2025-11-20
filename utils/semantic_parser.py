import json
from utils.api_client import client


def parse_with_llm(question):
    """
    Parse natural language into structured filtering conditions::
    {
        "Platform": str or None,
        "Price": {"min": float or None, "max": float or None, "cycle": "Monthly" or "Annual" or None},
        "Storage": {"min": float or None, "max": float or None},
        "Feature": str or None
    }
    """

    system_prompt = (
        "You are a semantic parser for a cloud storage QA system.\n"
        "Your output MUST ALWAYS be a JSON object with the following schema:\\n"
        "{\\n"
        '  "Platform": string|null,\\n'
        '  "Price": {"min": float|null, "max": float|null, "cycle": "Monthly"|"Annual"|null},\\n'
        '  "Storage": {"min": float|null, "max": float|null},\\n'
        '  "Feature": string|null\\n'
        "}\\n\\n"
        "=== PLATFORM EXTRACTION RULES ===\\n"
        "- Platforms: ['Google Drive','Dropbox','OneDrive','Box'].\\n"
        "- Only set Platform when the user clearly mentions it.\\n"
        "- If the user does NOT specify, return null.\\n"
        "- DO NOT guess. DO NOT default to Google Drive.\\n\\n"
        "=== PRICE EXTRACTION RULES ===\\n"
        "- For phrases like 'under 10', 'below 10', '<10', 'not exceeding 10', 'budget 10', 'within 10 dollars' → set Price.max = 10.\\n"
        "- For phrases like 'over 10', '>10', 'greater than 10' → set Price.min = 10.\\n"
        "- If the user says 'cheapest', 'lowest price', 'most affordable', you MAY leave min/max as null;\\n"
        "  the retriever will then choose the minimum price among candidates.\\n"
        "- If the user says 'most expensive', 'highest price', you MAY leave min/max as null;\\n"
        "  the retriever may handle this separately.\\n"
        "- If 'monthly' or 'per month' is mentioned → Price.cycle = 'Monthly'.\\n"
        "- If 'annual', 'yearly', 'per year', 'yearly payment' is mentioned → Price.cycle = 'Annual'.\\n"
        "- If no price constraint is mentioned → leave Price fields null.\\n\\n"
        "=== STORAGE EXTRACTION RULES ===\\n"
        "- Convert TB/GB mentioned by the user into numeric size in GB.\\n"
        "- For example, '2TB' → Storage.min = 2048 if user says 'at least 2TB', '>=2TB', 'no less than 2TB'.\\n"
        "- 'less than 1TB', '<1TB', 'smaller than 1TB' → Storage.max = 1024.\\n"
        "- If storage range is NOT specified → leave min/max null.\\n\\n"
        "=== FEATURE EXTRACTION RULES ===\\n"
        "- Extract ONE key feature phrase if the user clearly wants some capability, e.g. 'PDF', 'encryption', 'AI'.\\n"
        "- If the user does NOT request a feature → Feature = null.\\n\\n"
        "=== IMPORTANT RULES ===\\n"
        "- You MUST always return all four top-level fields.\\n"
        "- Any field not mentioned MUST be null.\\n"
        "- NO extra text. Output pure JSON only.\\n"
        "- DO NOT invent details.\\n"
    )

    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )

    raw = resp.choices[0].message.content.strip()
    print(f"DEBUG -raw semantic parse output: {raw}")

    # fallback prevent JSON parsing errors
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {
            "Platform": None,
            "Price": {"min": None, "max": None, "cycle": None},
            "Storage": {"min": None, "max": None},
            "Feature": None,
        }

    print(f"STEP 1- semantic parse result → {parsed}")
    return parsed
