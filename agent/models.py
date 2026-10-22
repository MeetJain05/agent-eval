import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter

# Locate and load root .env
project_root = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=project_root / ".env")

# Ensure GOOGLE_API_KEY is populated if GEMINI_API_KEY is provided
if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Groq rate limiter: conservative 20 RPM to avoid burst-rate 429s across
# evaluation test cases that each make multiple tool calls.
groq_rate_limiter = InMemoryRateLimiter(
    requests_per_second=20 / 60,   # ≈ 0.33 req/s
    check_every_n_seconds=0.1,
    max_bucket_size=10,
)

# Primary LLM: Groq (with 30 RPM rate limiter)
groq_api_key = os.getenv("GROQ_API_KEY")
groq_model_name = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

groq_llm = ChatGroq(
    model=groq_model_name,
    api_key=groq_api_key,
    rate_limiter=groq_rate_limiter,
    temperature=0,
)

# Secondary Fallback: Google Gemini
gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
gemini_model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

gemini_llm = ChatGoogleGenerativeAI(
    model=gemini_model_name,
    google_api_key=gemini_api_key,
    temperature=0,
)

# Hybrid Model: Primarily relies on Groq (rate-limited to 30 RPM), falls back to Gemini
model = groq_llm.with_fallbacks([gemini_llm])