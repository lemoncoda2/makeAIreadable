"""Shared utilities for the decoupled collaboration experiment."""

from .code_executor import (
    batch_compute_rewards,
    compute_reward,
    extract_code,
    execute_test,
    separate_output,
    safe_execute,
)
from .prompts import (
    CODING_SYSTEM_PROMPT,
    REGEN_SYSTEM_PROMPT,
    REGEN_USER_TEMPLATE,
    JUDGE_PROMPT,
    DPO_PROMPT_TEMPLATE,
    DPO_PROMPT_FORMAT,
    build_coding_messages,
    build_regen_messages,
    build_dpo_messages,
    build_dpo_prompt,
    render_dpo_prompt,
)
from .metrics import (
    parse_readability_score,
    aggregate_readability_scores,
    think_leak_rate,
    avg_length_tokens,
    syntax_error_rate,
    hypothesis_check,
    summarize_eval_results,
)
from .api_judge import (
    get_deepseek_client,
    parse_json_score,
    judge_collaboration,
    judge_collaboration_async,
    judge_batch,
    filter_pair,
    mock_judge_scores,
)

__all__ = [
    # code_executor
    "batch_compute_rewards",
    "compute_reward",
    "extract_code",
    "execute_test",
    "separate_output",
    "safe_execute",
    # prompts
    "CODING_SYSTEM_PROMPT",
    "REGEN_SYSTEM_PROMPT",
    "REGEN_USER_TEMPLATE",
    "JUDGE_PROMPT",
    "DPO_PROMPT_TEMPLATE",
    "DPO_PROMPT_FORMAT",
    "build_coding_messages",
    "build_regen_messages",
    "build_dpo_messages",
    "build_dpo_prompt",
    "render_dpo_prompt",
    # metrics
    "parse_readability_score",
    "aggregate_readability_scores",
    "think_leak_rate",
    "avg_length_tokens",
    "syntax_error_rate",
    "hypothesis_check",
    "summarize_eval_results",
    # api_judge
    "get_deepseek_client",
    "parse_json_score",
    "judge_collaboration",
    "judge_collaboration_async",
    "judge_batch",
    "filter_pair",
    "mock_judge_scores",
]
