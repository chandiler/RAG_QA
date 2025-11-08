from utils.api_client import client


def llm_only_answer(question):
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": question}]
    )
    return resp.choices[0].message.content.strip()
