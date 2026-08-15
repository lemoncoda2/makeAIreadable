"""Prompt templates and message builders for coding, regen, judge, and DPO."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

CODING_USER_SUFFIX = (
    "\n\nSolve this as a Python function with the exact interface implied by the "
    "prompt and examples. Think briefly and choose one approach. End with the "
    "complete runnable solution in exactly one fenced Python code block; code is "
    "mandatory and has priority over explanation."
)

REGEN_SYSTEM_PROMPT = """你是一个AI编程助手的"表达优化师"。
你的任务是：给定一个编程问题和已完成的解决方案，写一段简洁清晰的说明给用户。"""

REGEN_USER_TEMPLATE = """## 用户的编程需求
{task_prompt}

## 已完成的解决方案（思考过程）
{thinking}

## 已完成的代码
```python
{code}
```

## 请为用户写一段协作消息
要求：
1. 一句话说明你的理解
2. 一句话说明核心思路
3. 如果有重要的设计决策，简要说明
4. 不要重复代码内容
5. 总长度控制在50-150字（中文）或30-80词（英文）"""

JUDGE_PROMPT = """请评估以下AI编程助手回复的协作质量。

## 用户问题
{task_prompt}

## AI的回复
{collaboration_text}

## 评分标准（每项1-10分）
- clarity: 用户能否快速理解AI做了什么？
- conciseness: 是否简洁不冗余？
- informativeness: 是否传达了关键信息（思路、决策）？
- naturalness: 语言是否自然流畅？

请直接输出JSON（不要其他内容）:
{{"clarity": X, "conciseness": X, "informativeness": X, "naturalness": X, "overall": X}}"""

# Legacy fake-XML template — DO NOT use for training. Kept only so old callers
# fail loudly via build_dpo_prompt().
DPO_PROMPT_TEMPLATE = (
    "<system>你是编程助手。</system>\n"
    "<user>{task_prompt}</user>\n"
    "<think>{thinking}</think>\n"
    "代码已写好：\n"
    "```python\n"
    "{code}\n"
    "```\n"
    "请为用户写一段说明："
)

# Tag stored in filtered DPO jsonl metadata so train_dpo can re-render safely.
DPO_PROMPT_FORMAT = "qwen_chat_regen_v1"


def build_coding_messages(
    task_prompt: str,
    public_test_cases: Optional[Sequence[str]] = None,
) -> List[Dict[str, str]]:
    """Chat messages for coding generation with thinking mode."""
    # Qwen3 officially uses no default system message. Keeping the coding
    # contract in the user turn avoids pathological unclosed thinking loops seen
    # with a custom system role while enable_thinking=True remains explicit in
    # every caller.
    content = task_prompt.rstrip()
    if public_test_cases:
        public_example = str(public_test_cases[0]).strip()
        if public_example and public_example not in content:
            content += (
                "\n\nRequired public interface example (your definitions must "
                f"make this assertion executable):\n{public_example}"
            )
    return [{"role": "user", "content": content + CODING_USER_SUFFIX}]


def build_regen_messages(
    task_prompt: str,
    thinking: str,
    code: str,
) -> List[Dict[str, str]]:
    """Chat messages for regenerating collaboration text from a work trace."""
    user_content = REGEN_USER_TEMPLATE.format(
        task_prompt=task_prompt,
        thinking=thinking,
        code=code,
    )
    return [
        {"role": "system", "content": REGEN_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_dpo_messages(
    task_prompt: str, thinking: str, code: str
) -> List[Dict[str, str]]:
    """DPO conditioning messages — identical to regen (collaboration-only)."""
    return build_regen_messages(task_prompt, thinking, code)


def build_dpo_prompt(task_prompt: str, thinking: str, code: str) -> str:
    """Deprecated fake-XML string. Raises — use render_dpo_prompt with a tokenizer."""
    raise RuntimeError(
        "build_dpo_prompt() produced a fake <system>/<user> XML string that does not "
        "match Qwen chat templates used at inference/regen. "
        "Use build_dpo_messages() + apply_chat_template(..., enable_thinking=False), "
        "or utils.prompts.render_dpo_prompt(tokenizer, ...)."
    )


def render_dpo_prompt(tokenizer, task_prompt: str, thinking: str, code: str) -> str:
    """Render DPO prompt with the same chat template as regen_collaboration."""
    from utils.model_utils import apply_chat_template_with_thinking

    messages = build_dpo_messages(task_prompt, thinking, code)
    return apply_chat_template_with_thinking(
        tokenizer,
        messages,
        enable_thinking=False,
        add_generation_prompt=True,
        tokenize=False,
    )
