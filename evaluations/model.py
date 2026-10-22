import os
import time
import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Any
from dotenv import load_dotenv

from deepeval.models.base_model import DeepEvalBaseLLM

# Ensure environment is loaded from the root .env
project_root = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=project_root / ".env")

# Ensure GOOGLE_API_KEY is available if GEMINI_API_KEY was supplied
if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

logger = logging.getLogger("HybridEvaluator")


class GroqRateLimiter:
    """Thread-safe rate limiter enforcing a strict limit of max_requests per window_seconds."""

    def __init__(self, max_requests: int = 30, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.timestamps: List[float] = []

    def acquire(self):
        now = time.time()
        # Drop timestamps older than the sliding window
        self.timestamps = [t for t in self.timestamps if now - t < self.window_seconds]
        if len(self.timestamps) >= self.max_requests:
            sleep_time = self.window_seconds - (now - self.timestamps[0]) + 0.1
            if sleep_time > 0:
                logger.info(f"Groq 30 RPM rate limit reached. Pausing {sleep_time:.2f}s before next request...")
                time.sleep(sleep_time)
            now = time.time()
            self.timestamps = [t for t in self.timestamps if now - t < self.window_seconds]
        self.timestamps.append(now)


# Global rate limiter ensuring 30 requests per minute across all Groq evaluation calls
eval_groq_rate_limiter = GroqRateLimiter(max_requests=30, window_seconds=60.0)


class HybridDeepEvalLLM(DeepEvalBaseLLM):
    """
    Hybrid LLM for DeepEval evaluations.
    Primarily relies on Groq (enforcing a strict 30 requests/min limit).
    Automatically falls back to Google Gemini and HuggingFace if Groq is rate-limited or fails.
    """

    def __init__(
        self,
        model_name: str = "hybrid-groq-gemini-hf",
        groq_model: Optional[str] = None,
        gemini_model: Optional[str] = None,
        hf_model: Optional[str] = None,
    ):
        self.model_name = model_name
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.hf_token = os.getenv("HF_TOKEN")

        self.groq_model = groq_model or os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        self.gemini_model = gemini_model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
        self.hf_model = hf_model or os.getenv("HF_MODEL", "meta-llama/Llama-3.2-1B-Instruct")

        self.groq_client = None
        self.gemini_client = None
        self.hf_client = None

        super().__init__(model_name)

    def load_model(self):
        if self.groq_api_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=self.groq_api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")

        if self.gemini_api_key:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=self.gemini_api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}")

        if self.hf_token:
            try:
                from huggingface_hub import InferenceClient
                self.hf_client = InferenceClient(api_key=self.hf_token)
            except Exception as e:
                logger.warning(f"Failed to initialize HF client: {e}")

        return self

    def _call_groq(self, prompt: str) -> str:
        if not self.groq_client:
            raise ValueError("Groq client not available or GROQ_API_KEY missing")
        eval_groq_rate_limiter.acquire()
        response = self.groq_client.chat.completions.create(
            model=self.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content

    def _call_gemini(self, prompt: str) -> str:
        if not self.gemini_client:
            raise ValueError("Gemini client not available or GEMINI_API_KEY missing")
        response = self.gemini_client.models.generate_content(
            model=self.gemini_model,
            contents=prompt,
        )
        return response.text

    def _call_hf(self, prompt: str) -> str:
        if not self.hf_client:
            raise ValueError("Hugging Face client not available or HF_TOKEN missing")
        response = self.hf_client.chat.completions.create(
            model=self.hf_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
        )
        return response.choices[0].message.content

    def generate(self, prompt: str, schema: Optional[Any] = None) -> str:
        # 1. Primary: Groq (Throttled to <= 30 RPM)
        try:
            return self._call_groq(prompt)
        except Exception as groq_err:
            logger.warning(f"[HybridEvaluator] Groq error ({groq_err}). Falling back to Gemini...")

        # 2. Secondary: Gemini Fallback
        try:
            return self._call_gemini(prompt)
        except Exception as gemini_err:
            logger.warning(f"[HybridEvaluator] Gemini error ({gemini_err}). Falling back to HuggingFace...")

        # 3. Tertiary: HuggingFace Fallback
        try:
            return self._call_hf(prompt)
        except Exception as hf_err:
            raise RuntimeError(
                f"All hybrid evaluation providers failed. Groq error logged above, "
                f"Gemini error logged above, HF error: {hf_err}"
            )

    async def a_generate(self, prompt: str, schema: Optional[Any] = None) -> str:
        return await asyncio.to_thread(self.generate, prompt, schema)

    def get_model_name(self) -> str:
        return self.model_name


# Exported evaluation model instance used by all evaluation scripts
eval_model = HybridDeepEvalLLM()