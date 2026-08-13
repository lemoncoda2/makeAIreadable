"""Evaluation metrics: readability aggregation, leak rate, hypothesis checks."""

from __future__ import annotations

import ast
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union


SCORE_KEYS = ("clarity", "conciseness", "informativeness", "naturalness", "overall")


def parse_readability_score(score: Union[Mapping[str, Any], str, None]) -> Dict[str, float]:
    """Normalize a readability score dict (or JSON-like string) to floats.

    Missing dimensions default to ``0.0``. If ``overall`` is absent, it is the
    mean of the four detail dimensions when any are present.
    """
    if score is None:
        return {k: 0.0 for k in SCORE_KEYS}

    if isinstance(score, str):
        from .api_judge import parse_json_score

        score = parse_json_score(score)

    result: Dict[str, float] = {}
    for key in SCORE_KEYS:
        val = score.get(key) if isinstance(score, Mapping) else None
        try:
            result[key] = float(val) if val is not None else 0.0
        except (TypeError, ValueError):
            result[key] = 0.0

    if (
        isinstance(score, Mapping)
        and "overall" not in score
        and any(score.get(k) is not None for k in SCORE_KEYS[:-1])
    ):
        details = [result[k] for k in SCORE_KEYS[:-1]]
        result["overall"] = sum(details) / len(details)

    return result


def aggregate_readability_scores(
    scores: Sequence[Mapping[str, Any]],
) -> Dict[str, float]:
    """Mean readability scores across a list of score dicts."""
    if not scores:
        return {k: 0.0 for k in SCORE_KEYS}

    parsed = [parse_readability_score(s) for s in scores]
    n = len(parsed)
    return {k: sum(p[k] for p in parsed) / n for k in SCORE_KEYS}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def think_leak_rate(
    thinking: str,
    collaboration: str,
    n: int = 3,
) -> float:
    """Fraction of thinking n-grams that also appear in collaboration text.

    Returns ``0.0`` when thinking has no n-grams. Higher values mean more
    reasoning content leaked into the user-facing response.
    """
    think_tokens = _tokenize(thinking or "")
    collab_tokens = _tokenize(collaboration or "")

    if len(think_tokens) < n:
        return 0.0

    def ngrams(tokens: List[str]) -> List[tuple]:
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    think_ngrams = ngrams(think_tokens)
    if not think_ngrams:
        return 0.0

    collab_set = set(ngrams(collab_tokens))
    leaked = sum(1 for g in think_ngrams if g in collab_set)
    return leaked / len(think_ngrams)


def avg_length_tokens(texts: Iterable[str]) -> float:
    """Average whitespace-delimited token count over ``texts``."""
    texts_list = list(texts)
    if not texts_list:
        return 0.0
    lengths = [len((t or "").split()) for t in texts_list]
    return sum(lengths) / len(lengths)


def syntax_error_rate(codes: Sequence[str]) -> float:
    """Fraction of code strings that fail ``ast.parse``."""
    if not codes:
        return 0.0

    errors = 0
    for code in codes:
        try:
            ast.parse(code or "")
        except SyntaxError:
            errors += 1
    return errors / len(codes)


def _fmt_delta_pct(delta: float) -> str:
    pct = delta * 100.0
    sign = "+" if pct >= 0 else ""
    if abs(pct - round(pct)) < 1e-9:
        return f"{sign}{int(round(pct))}%"
    return f"{sign}{pct:.1f}%"


def _fmt_delta_abs(delta: float) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}"


def hypothesis_check(
    base_metrics: Mapping[str, Any],
    rl_metrics: Mapping[str, Any],
    final_metrics: Optional[Mapping[str, Any]] = None,
    *,
    coding_improve_threshold: float = 0.05,
    coding_preserve_gap: float = 0.02,
) -> Dict[str, Dict[str, Any]]:
    """Verify H1–H4 from GOAL success criteria.

    Expected metric keys (flexible aliases accepted):
    - coding: ``mbpp_plus_pass1`` / ``pass_rate`` / ``pass_at_1``
    - readability: ``readability_overall`` / ``overall`` / ``readability``

    H1: RL improves coding vs base by ≥ ``coding_improve_threshold`` (default 5%).
    H2: RL hurts readability vs base (rl < base).
    H3: DPO recovers readability to ≥ base (requires ``final_metrics``).
    H4: DPO preserves coding within ``coding_preserve_gap`` of RL (default 2%).
    """

    def _coding(m: Mapping[str, Any]) -> float:
        for key in ("mbpp_plus_pass1", "pass_rate", "pass_at_1", "coding"):
            if key in m and m[key] is not None:
                return float(m[key])
        raise KeyError(
            "coding metric missing; expected one of "
            "mbpp_plus_pass1 / pass_rate / pass_at_1 / coding"
        )

    def _readability(m: Mapping[str, Any]) -> float:
        for key in ("readability_overall", "overall", "readability"):
            if key in m and m[key] is not None:
                return float(m[key])
        raise KeyError(
            "readability metric missing; expected one of "
            "readability_overall / overall / readability"
        )

    base_code = _coding(base_metrics)
    rl_code = _coding(rl_metrics)
    base_read = _readability(base_metrics)
    rl_read = _readability(rl_metrics)

    h1_delta = rl_code - base_code
    h2_delta = rl_read - base_read

    results: Dict[str, Dict[str, Any]] = {
        "H1_rl_improves_coding": {
            "verified": h1_delta >= coding_improve_threshold,
            "delta": _fmt_delta_pct(h1_delta),
            "base": base_code,
            "rl": rl_code,
        },
        "H2_rl_hurts_readability": {
            "verified": rl_read < base_read,
            "delta": _fmt_delta_abs(h2_delta),
            "base": base_read,
            "rl": rl_read,
        },
    }

    if final_metrics is None:
        results["H3_dpo_recovers_readability"] = {
            "verified": False,
            "delta": None,
            "skipped": True,
            "reason": "final_metrics not provided",
        }
        results["H4_dpo_preserves_coding"] = {
            "verified": False,
            "delta": None,
            "skipped": True,
            "reason": "final_metrics not provided",
        }
        return results

    final_code = _coding(final_metrics)
    final_read = _readability(final_metrics)

    h3_delta = final_read - rl_read
    h4_delta = final_code - rl_code

    results["H3_dpo_recovers_readability"] = {
        "verified": final_read >= base_read,
        "delta": _fmt_delta_abs(h3_delta),
        "base": base_read,
        "rl": rl_read,
        "final": final_read,
    }
    results["H4_dpo_preserves_coding"] = {
        "verified": abs(final_code - rl_code) <= coding_preserve_gap,
        "delta": _fmt_delta_pct(h4_delta),
        "rl": rl_code,
        "final": final_code,
    }
    return results


def summarize_eval_results(
    *,
    cycle: int = 0,
    models: Optional[Mapping[str, Mapping[str, Any]]] = None,
    base_metrics: Optional[Mapping[str, Any]] = None,
    rl_metrics: Optional[Mapping[str, Any]] = None,
    final_metrics: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the evaluate.py JSON result shape from GOAL Step 4.3.

    Prefer passing a ``models`` mapping keyed by ``base`` / ``rl`` / ``final``.
    Alternatively pass flat ``*_metrics`` dicts which are nested under ``models``.
    """
    model_block: Dict[str, Any] = {}
    if models:
        model_block = {k: dict(v) for k, v in models.items()}
    else:
        if base_metrics is not None:
            model_block["base"] = dict(base_metrics)
        if rl_metrics is not None:
            model_block["rl"] = dict(rl_metrics)
        if final_metrics is not None:
            model_block["final"] = dict(final_metrics)

    hyp_base = model_block.get("base")
    hyp_rl = model_block.get("rl")
    hyp_final = model_block.get("final")

    hypothesis_results: Dict[str, Any] = {}
    if hyp_base is not None and hyp_rl is not None:
        hypothesis_results = hypothesis_check(hyp_base, hyp_rl, hyp_final)

    result: Dict[str, Any] = {
        "cycle": cycle,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "models": model_block,
        "hypothesis_results": hypothesis_results,
    }
    if extra:
        result.update(extra)
    return result
