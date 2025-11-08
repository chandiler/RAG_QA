from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ No OPENAI_API_KEY found in .env")

client = OpenAI(api_key=api_key)
