"""Qwen3 thinking-budget generation helpers."""

from __future__ import annotations

from typing import Any


def qwen_think_end_token_id(tokenizer) -> int:
    """Return Qwen3's single-token ``</think>`` id or fail fast."""
    token_ids = tokenizer.encode("</think>", add_special_tokens=False)
    if len(token_ids) != 1:
        raise RuntimeError(
            "Qwen thinking budget requires </think> to encode as exactly one "
            f"token; got {token_ids!r}"
        )
    return int(token_ids[0])


def qwen_fence_token_id(tokenizer) -> int:
    """Return the standalone Markdown triple-backtick token or fail fast."""
    token_ids = tokenizer.encode("```", add_special_tokens=False)
    if len(token_ids) != 1:
        raise RuntimeError(
            "Code-fence stopping requires ``` to encode as exactly one token; "
            f"got {token_ids!r}"
        )
    return int(token_ids[0])


def build_code_fence_stopping_criteria(tokenizer, *, prompt_length: int):
    """Stop each row after its post-thinking fenced Python block closes.

    Compares token ids only. ``</think>`` and ````` must each be one token.
    """
    from transformers import StoppingCriteria, StoppingCriteriaList

    think_end_id = qwen_think_end_token_id(tokenizer)
    fence_id = qwen_fence_token_id(tokenizer)

    class CodeFenceStoppingCriteria(StoppingCriteria):
        def __init__(self):
            self.processed_length = 0
            self.after_think: list[bool] = []
            self.fence_counts: list[int] = []

        def __call__(self, input_ids, scores, **kwargs):
            generated = input_ids[:, prompt_length:]
            batch_size = int(generated.shape[0])
            if not self.after_think:
                self.after_think = [False] * batch_size
                self.fence_counts = [0] * batch_size
            if len(self.after_think) != batch_size:
                raise RuntimeError("Code-fence stopping batch size changed mid-generation")

            for row in range(batch_size):
                for token in generated[row, self.processed_length :].tolist():
                    tid = int(token)
                    if tid == think_end_id:
                        self.after_think[row] = True
                        self.fence_counts[row] = 0
                        continue
                    if self.after_think[row] and tid == fence_id:
                        self.fence_counts[row] += 1
            self.processed_length = int(generated.shape[1])
            return generated.new_tensor(
                [count >= 2 for count in self.fence_counts], dtype=bool
            )

    return StoppingCriteriaList([CodeFenceStoppingCriteria()])


def mask_completion_after_code_fence(completion_ids, completion_mask, tokenizer):
    """Mask tokens after the second post-thinking Markdown fence (token-id only)."""
    import torch

    think_end_id = qwen_think_end_token_id(tokenizer)
    fence_id = qwen_fence_token_id(tokenizer)
    trimmed = completion_mask.clone()
    for row in range(int(completion_ids.shape[0])):
        after_think = False
        fence_count = 0
        close_index = None
        for index, token in enumerate(completion_ids[row].tolist()):
            tid = int(token)
            if tid == think_end_id:
                after_think = True
                fence_count = 0
                continue
            if not after_think:
                continue
            if tid == fence_id:
                fence_count += 1
                if fence_count >= 2:
                    close_index = index
                    break
        if close_index is not None and close_index + 1 < trimmed.shape[1]:
            trimmed[row, close_index + 1 :] = torch.zeros_like(
                trimmed[row, close_index + 1 :]
            )
    return trimmed


def build_thinking_budget_logits_processor(
    tokenizer,
    *,
    prompt_length: int,
    thinking_budget_tokens: int,
):
    """Force ``</think>`` once when an open thought reaches its token budget.

    Rows that naturally emitted ``</think>`` before the budget are untouched.
    The returned processor is suitable for ``transformers.generate``.
    """
    if thinking_budget_tokens <= 0:
        raise ValueError("thinking_budget_tokens must be positive")

    import torch
    from transformers import LogitsProcessor, LogitsProcessorList

    think_end_id = qwen_think_end_token_id(tokenizer)

    class ThinkingBudgetLogitsProcessor(LogitsProcessor):
        def __call__(self, input_ids, scores):
            generated = input_ids[:, prompt_length:]
            if generated.shape[1] != thinking_budget_tokens:
                return scores

            needs_close = ~generated.eq(think_end_id).any(dim=1)
            if not bool(needs_close.any()):
                return scores

            scores = scores.clone()
            forced = torch.full_like(scores[needs_close], float("-inf"))
            forced[:, think_end_id] = 0.0
            scores[needs_close] = forced
            return scores

    return LogitsProcessorList([ThinkingBudgetLogitsProcessor()])


def install_thinking_budget_generate(
    model: Any,
    tokenizer: Any,
    *,
    thinking_budget_tokens: int,
    stop_after_code_fence: bool = False,
) -> int:
    """Wrap one model instance's ``generate`` with the budget processor."""
    if getattr(model, "_decoupled_thinking_budget_installed", False):
        raise RuntimeError("thinking-budget generate wrapper is already installed")
    if not callable(getattr(model, "generate", None)):
        raise TypeError("model has no callable generate method")

    think_end_id = qwen_think_end_token_id(tokenizer)
    original_generate = model.generate

    def budgeted_generate(*args, **kwargs):
        if kwargs.get("logits_processor") is not None:
            raise RuntimeError(
                "Refusing to combine an unknown logits_processor with the "
                "Qwen thinking-budget controller"
            )
        if stop_after_code_fence and kwargs.get("stopping_criteria") is not None:
            raise RuntimeError(
                "Refusing to combine unknown stopping_criteria with the "
                "code-fence controller"
            )
        if args:
            input_ids = args[0]
        else:
            input_ids = kwargs.get("input_ids")
        if input_ids is None:
            raise RuntimeError("thinking-budget generation requires input_ids")
        kwargs["logits_processor"] = build_thinking_budget_logits_processor(
            tokenizer,
            prompt_length=int(input_ids.shape[1]),
            thinking_budget_tokens=thinking_budget_tokens,
        )
        if stop_after_code_fence:
            kwargs["stopping_criteria"] = build_code_fence_stopping_criteria(
                tokenizer,
                prompt_length=int(input_ids.shape[1]),
            )
        return original_generate(*args, **kwargs)

    model.generate = budgeted_generate
    model._decoupled_thinking_budget_installed = True
    model._decoupled_thinking_budget_tokens = thinking_budget_tokens
    model._decoupled_stop_after_code_fence = stop_after_code_fence
    model._decoupled_think_end_token_id = think_end_id
    return think_end_id
