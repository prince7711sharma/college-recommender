"""
LLM Service — sends filtered colleges + student profile to Groq for intelligent ranking.

Resilience features:
  • Exponential back-off with jitter on 429 rate-limit responses.
  • Structured error dicts returned (never raised) for every failure path.
  • In-memory TTL cache to avoid redundant API calls.
  • Automatic fallback to a cheaper model on primary quota exhaustion.
  • Static mock response used as last resort so the API never crashes.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration  (set these as environment variables on Render / locally in .env)
# ---------------------------------------------------------------------------
GROQ_API_URL: str = os.getenv(
    "GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions"
)
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
PRIMARY_MODEL: str = os.getenv("LLM_PRIMARY_MODEL", "llama-3.3-70b-versatile")
FALLBACK_MODEL: str = os.getenv("LLM_FALLBACK_MODEL", "llama3-8b-8192")
MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
BASE_BACKOFF: float = float(os.getenv("LLM_BACKOFF_BASE", "2.0"))
CACHE_TTL: int = int(os.getenv("LLM_CACHE_TTL", "1800"))  # seconds

# ---------------------------------------------------------------------------
# In-memory cache  (swap for Redis on multi-instance deployments)
# ---------------------------------------------------------------------------
_RESPONSE_CACHE: Dict[str, Dict[str, Any]] = {}


def _cache_key(prompt: str, model: str) -> str:
    return f"{model}:{hash(prompt)}"


def _get_cached(prompt: str, model: str) -> Optional[Dict[str, Any]]:
    key = _cache_key(prompt, model)
    entry = _RESPONSE_CACHE.get(key)
    if entry and (time.time() - entry["ts"]) < CACHE_TTL:
        return entry["response"]
    _RESPONSE_CACHE.pop(key, None)
    return None


def _store_cache(prompt: str, model: str, response: Dict[str, Any]) -> None:
    _RESPONSE_CACHE[_cache_key(prompt, model)] = {
        "ts": time.time(),
        "response": response,
    }


# ---------------------------------------------------------------------------
# Static mock — used when BOTH primary and fallback models fail.
# This guarantees the API always returns a well-formed JSON structure.
# ---------------------------------------------------------------------------
_MOCK_CONTENT = json.dumps(
    {
        "student_summary": (
            "The AI recommendation engine is temporarily unavailable. "
            "Please try again in a few minutes."
        ),
        "recommendations": [],
        "notice": "This is a cached/mock response. The LLM service is down.",
        "_is_mock": True,
    },
    ensure_ascii=False,
)
MOCK_FALLBACK_RESPONSE: Dict[str, Any] = {
    "choices": [{"message": {"content": _MOCK_CONTENT}}]
}


# ---------------------------------------------------------------------------
# Low-level HTTP helper
# ---------------------------------------------------------------------------
def _request_llm(prompt: str, model: str, retry: int = 0, max_tokens: int = 2048) -> Dict[str, Any]:
    """Call Groq and return the raw JSON dict.

    On failure, always returns a structured error dict — never raises.
    """
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY is not set.")
        return {
            "error": "Configuration Error",
            "detail": "GROQ_API_KEY environment variable is missing.",
            "status_code": 503,
        }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }

    try:
        resp = httpx.post(GROQ_API_URL, headers=headers, json=payload, timeout=30.0)
        resp.raise_for_status()
        return resp.json()

    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code

        # -- 429 : rate-limited → retry with exponential back-off + jitter --
        if status == 429:
            if retry >= MAX_RETRIES:
                logger.error(
                    "Rate limit: giving up after %s retries (model=%s).", retry, model
                )
                return {
                    "error": "AI Service Unavailable",
                    "detail": "Groq rate limit exceeded. Please try again later.",
                    "status_code": 503,
                }
            backoff = BASE_BACKOFF * (2 ** retry) + random.uniform(0.1, 0.9)
            logger.warning(
                "Rate limit hit (model=%s). Back-off %.2f s (retry %s/%s).",
                model,
                backoff,
                retry + 1,
                MAX_RETRIES,
            )
            time.sleep(backoff)
            return _request_llm(prompt, model, retry + 1, max_tokens)

        # -- Any other HTTP error --
        logger.error(
            "Groq HTTP error %s for model=%s: %s", status, model, exc.response.text[:200]
        )
        return {
            "error": "AI Service Unavailable",
            "detail": f"Groq returned HTTP {status}.",
            "status_code": 503,
        }

    except (httpx.ConnectError, httpx.TimeoutException, OSError) as exc:
        logger.error("Network error contacting Groq (model=%s): %s", model, exc)
        return {
            "error": "AI Service Unavailable",
            "detail": "Cannot reach Groq API. Check network / DNS.",
            "status_code": 503,
        }

    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error contacting Groq (model=%s): %s", model, exc)
        return {
            "error": "AI Service Unavailable",
            "detail": str(exc),
            "status_code": 503,
        }


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------
def get_completion(prompt: str) -> Dict[str, Any]:
    """Return a Groq response dict that always contains a ``choices`` key.

    Priority:
      1. Cache (primary model)
      2. Live call — primary model
      3. Cache (fallback model)
      4. Live call — fallback model
      5. Static mock response (so the API never returns a crash)
    """
    # 1. Cache – primary
    cached = _get_cached(prompt, PRIMARY_MODEL)
    if cached:
        logger.debug("Cache hit (primary).")
        return cached

    # 2. Live – primary
    resp = _request_llm(prompt, PRIMARY_MODEL)
    if isinstance(resp, dict) and "choices" in resp:
        _store_cache(prompt, PRIMARY_MODEL, resp)
        return resp
    logger.warning(
        "Primary model failed (%s). Trying fallback.", resp.get("detail", "unknown")
    )

    # 3. Cache – fallback
    cached_fb = _get_cached(prompt, FALLBACK_MODEL)
    if cached_fb:
        logger.debug("Cache hit (fallback).")
        return cached_fb

    # 4. Live – fallback (use smaller max_tokens to stay under the 6000 TPM limit)
    resp_fb = _request_llm(prompt, FALLBACK_MODEL, max_tokens=1200)
    if isinstance(resp_fb, dict) and "choices" in resp_fb:
        _store_cache(prompt, FALLBACK_MODEL, resp_fb)
        return resp_fb
    logger.error(
        "Fallback model also failed (%s). Using static mock.",
        resp_fb.get("detail", "unknown"),
    )

    # 5. Static mock — guaranteed well-formed response
    return MOCK_FALLBACK_RESPONSE


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------
class LLMService:
    """High-level interface for college recommendation prompts."""

    def __init__(self) -> None:
        if not GROQ_API_KEY:
            logger.warning(
                "GROQ_API_KEY is not set — LLM calls will fail and use the mock fallback."
            )
        self.model = PRIMARY_MODEL

    # ------------------------------------------------------------------
    def get_recommendations(
        self,
        student: Dict[str, Any],
        filtered_colleges: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build prompt → call LLM → parse JSON.  Always returns a dict."""

        # Keep only essential fields + limit to top 8 colleges to stay under token limits.
        # The filter already returns the highest-scored colleges so we lose nothing important.
        KEEP_FIELDS = {
            "name", "city", "state", "type", "courses", "branches",
            "fees_per_year", "entrance_exam", "cutoff_rank",
            "placement_avg_lpa", "naac_grade", "hostel", "website",
        }
        clean_colleges = [
            {k: v for k, v in c.items() if k in KEEP_FIELDS}
            for c in filtered_colleges[:8]   # top 8 is plenty for the LLM
        ]
        college_data = json.dumps(clean_colleges, indent=2, default=str)

        prompt = f"""You are an expert Indian career counselor and college admission advisor.

A student has the following profile:
- Marks: {student.get('marks')}%
- Annual Budget: ₹{student.get('budget', 0):,}
- Preferred Location: {student.get('location')}
- Desired Course: {student.get('course')}
- Preferred Branch: {student.get('branch', 'Any')}
- Category: {student.get('category', 'General')}

Below is the ONLY college data you may use. Do NOT invent or hallucinate any college.
You MUST pick ONLY from this list:

{college_data}

INSTRUCTIONS:
1. Select the top 3 to 5 best-fit colleges from the above list.
2. Rank them from best to least fit.
3. For each, explain WHY it suits this student (fees, location, placements, course match).
4. Return ONLY valid JSON in this exact format — no markdown, no backticks:

{{
  "student_summary": "Brief summary of the student profile and needs",
  "recommendations": [
    {{
      "name": "College Name",
      "city": "City",
      "state": "State",
      "type": "Government/Private",
      "fees_per_year": "₹X,XX,XXX",
      "branches_available": ["Branch1", "Branch2"],
      "entrance_exam": ["Exam1"],
      "placement_avg_lpa": 10,
      "naac_grade": "A+",
      "hostel": true,
      "website": "www.example.com",
      "reason": "Detailed reason why this college suits the student"
    }}
  ]
}}"""

        # --- call the LLM ---
        llm_resp = get_completion(prompt)

        # Extract text content from the choices structure
        try:
            content: str = llm_resp["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("Malformed LLM response structure: %s | %s", exc, llm_resp)
            return {
                "error": "AI Service Unavailable",
                "detail": "Malformed response from LLM service.",
                "recommendations": [],
                "status_code": 503,
            }

        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.splitlines()
            content = "\n".join(lines[1:] if lines[-1].strip() == "```" else lines[1:]).rstrip("`").strip()

        # Parse JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.exception("LLM returned non-JSON content: %s", content[:300])
            return {
                "error": "LLM returned invalid JSON",
                "detail": "The recommendation engine returned unreadable data.",
                "recommendations": [],
            }
