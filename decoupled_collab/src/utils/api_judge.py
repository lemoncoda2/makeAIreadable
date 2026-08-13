"""DeepSeek OpenAI-compatible client for collaboration readability judging."""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .prompts import JUDGE_PROMPT

# Re-export for callers that import JUDGE_PROMPT from this module (GOAL listing).
__all__ = [
    "JUDGE_PROMPT",
    "get_deepseek_client",
    "parse_json_score",
    "judge_collaboration",
    "judge_collaboration_async",
    "judge_batch",
    "filter_pair",
    "mock_judge_scores",
]

SCORE_KEYS = ("clarity", "conciseness", "informativeness", "naturalness", "overall")

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


def get_deepseek_client(api_key: Optional[str] = None, base_url: Optional[str] = None):
    """Create a sync OpenAI client pointed at DeepSeek.

    Env:
        ``DEEPSEEK_API_KEY`` (required unless ``api_key`` given)
        ``DEEPSEEK_BASE_URL`` (default ``https://api.deepseek.com``)
    """
    from openai import OpenAI

    key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError(
            "DEEPSEEK_API_KEY is not set. Pass api_key= or set the environment variable."
        )
    url = base_url or os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)
    return OpenAI(api_key=key, base_url=url)


def get_deepseek_async_client(
    api_key: Optional[str] = None, base_url: Optional[str] = None
):
    """Create an async OpenAI client pointed at DeepSeek."""
    from openai import AsyncOpenAI

    key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError(
            "DEEPSEEK_API_KEY is not set. Pass api_key= or set the environment variable."
        )
    url = base_url or os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)
    return AsyncOpenAI(api_key=key, base_url=url)


def _model_name() -> str:
    return os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL)


def parse_json_score(text: str) -> Dict[str, float]:
    """Robustly extract a readability score JSON object from model text.

    Handles fenced code blocks, surrounding prose, and trailing commas.
    Missing keys default to ``0.0``. If ``overall`` is missing, averages the
    four detail scores when any are present.
    """
    empty = {k: 0.0 for k in SCORE_KEYS}
    if not text or not str(text).strip():
        return empty

    raw = str(text).strip()

    # Prefer fenced JSON / generic fences
    fence = re.search(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if fence:
        raw = fence.group(1).strip()

    candidates: List[str] = [raw]

    # Extract outermost {...} spans
    brace_matches = list(re.finditer(r"\{[^{}]*\}", raw, re.DOTALL))
    if not brace_matches:
        # Nested braces: find first { to last }
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            candidates.append(raw[start : end + 1])
    else:
        for m in brace_matches:
            candidates.append(m.group(0))

    parsed: Optional[dict] = None
    for cand in candidates:
        cleaned = re.sub(r",\s*}", "}", cand)
        cleaned = re.sub(r",\s*]", "]", cleaned)
        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            parsed = obj
            break

    if parsed is None:
        # Last resort: key: number pairs
        found: Dict[str, float] = {}
        for key in SCORE_KEYS:
            m = re.search(
                rf'["\']?{key}["\']?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)',
                text,
                re.IGNORECASE,
            )
            if m:
                found[key] = float(m.group(1))
        if not found:
            return empty
        parsed = found

    result: Dict[str, float] = {}
    for key in SCORE_KEYS:
        val = parsed.get(key)
        try:
            result[key] = float(val) if val is not None else 0.0
        except (TypeError, ValueError):
            result[key] = 0.0

    if "overall" not in parsed and any(
        parsed.get(k) is not None for k in SCORE_KEYS[:-1]
    ):
        details = [result[k] for k in SCORE_KEYS[:-1]]
        result["overall"] = sum(details) / len(details)

    return result


def _build_judge_messages(task_prompt: str, text: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "user",
            "content": JUDGE_PROMPT.format(
                task_prompt=task_prompt,
                collaboration_text=text,
            ),
        }
    ]


def judge_collaboration(
    task_prompt: str,
    text: str,
    client: Any = None,
    *,
    model: Optional[str] = None,
) -> Dict[str, float]:
    """Synchronously score collaboration text via DeepSeek."""
    client = client or get_deepseek_client()
    model_name = model or _model_name()
    resp = client.chat.completions.create(
        model=model_name,
        messages=_build_judge_messages(task_prompt, text),
        max_tokens=100,
        temperature=0.0,
    )
    content = resp.choices[0].message.content or ""
    return parse_json_score(content)


async def judge_collaboration_async(
    task_prompt: str,
    text: str,
    client: Any = None,
    *,
    model: Optional[str] = None,
    max_retries: int = 5,
    base_delay: float = 1.0,
) -> Dict[str, float]:
    """Async score with exponential backoff on HTTP 429 / rate limits."""
    owns_client = client is None
    client = client or get_deepseek_async_client()
    model_name = model or _model_name()

    delay = base_delay
    last_exc: Optional[BaseException] = None
    try:
        for attempt in range(max_retries + 1):
            try:
                resp = await client.chat.completions.create(
                    model=model_name,
                    messages=_build_judge_messages(task_prompt, text),
                    max_tokens=100,
                    temperature=0.0,
                )
                content = resp.choices[0].message.content or ""
                return parse_json_score(content)
            except Exception as exc:  # noqa: BLE001 — retry classification below
                last_exc = exc
                status = getattr(exc, "status_code", None) or getattr(
                    getattr(exc, "response", None), "status_code", None
                )
                msg = str(exc).lower()
                is_rate_limit = status == 429 or "429" in msg or "rate limit" in msg
                if not is_rate_limit or attempt >= max_retries:
                    raise
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60.0)
        if last_exc:
            raise last_exc
        return {k: 0.0 for k in SCORE_KEYS}
    finally:
        if owns_client:
            close = getattr(client, "close", None)
            if close is not None:
                result = close()
                if asyncio.iscoroutine(result):
                    await result


async def judge_batch(
    items: Sequence[Mapping[str, str]],
    threshold: Optional[float] = None,
    max_concurrent: int = 5,
    *,
    client: Any = None,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Score many ``{task_prompt, text}`` items with bounded concurrency.

    If ``threshold`` is set, each result includes ``passes_threshold`` based on
    ``overall``.
    """
    owns_client = client is None
    client = client or get_deepseek_async_client()
    sem = asyncio.Semaphore(max_concurrent)

    async def _one(item: Mapping[str, str]) -> Dict[str, Any]:
        async with sem:
            score = await judge_collaboration_async(
                item["task_prompt"],
                item["text"],
                client=client,
                model=model,
            )
            out: Dict[str, Any] = {"score": score, **dict(item)}
            if threshold is not None:
                out["passes_threshold"] = score.get("overall", 0.0) >= threshold
            return out

    try:
        return list(await asyncio.gather(*[_one(it) for it in items]))
    finally:
        if owns_client:
            close = getattr(client, "close", None)
            if close is not None:
                result = close()
                if asyncio.iscoroutine(result):
                    await result


def filter_pair(
    regen_score: Mapping[str, Any],
    rl_score: Mapping[str, Any],
    threshold: float = 6.0,
    min_gap: float = 0.5,
) -> bool:
    """Return True if the regen/RL pair is usable for DPO.

    Conditions (GOAL Step 3.2):
    1. regen overall ≥ threshold
    2. regen overall > rl overall
    3. regen - rl ≥ min_gap
    """
    try:
        regen_overall = float(regen_score["overall"])
        rl_overall = float(rl_score["overall"])
    except (KeyError, TypeError, ValueError):
        return False

    if regen_overall < threshold:
        return False
    if regen_overall <= rl_overall:
        return False
    if regen_overall - rl_overall < min_gap:
        return False
    return True


def mock_judge_scores(
    regen_text: str,
    rl_text: str,
    *,
    base_regen: float = 7.5,
    base_rl: float = 5.0,
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Offline synthetic scores: regen > rl when texts differ (else equal)."""

    def _pack(overall: float) -> Dict[str, float]:
        return {
            "clarity": overall,
            "conciseness": max(0.0, overall - 0.2),
            "informativeness": max(0.0, overall - 0.1),
            "naturalness": overall,
            "overall": overall,
        }

    differs = (regen_text or "").strip() != (rl_text or "").strip()
    if differs:
        return _pack(base_regen), _pack(base_rl)
    return _pack(5.0), _pack(5.0)
