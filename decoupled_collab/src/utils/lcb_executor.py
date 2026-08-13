"""LiveCodeBench-style code execution (stdin + call-based).

Inspired by the official harness:
https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/evaluation/testing_util.py

We do **not** require installing the full ``lcb_runner`` package. Tests are stored
on each task as structured ``lcb_tests`` (see ``prepare_data.prepare_lcb_easy``).
This replaces the incorrect approach of stuffing JSON blobs into assert-only
``code_executor.execute_test``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Union


def normalize_stdout(text: str) -> str:
    """Strip trailing whitespace per line; drop trailing empty lines."""
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def parse_public_test_cases(raw: Any) -> List[Dict[str, Any]]:
    """Normalize HF/LCB ``public_test_cases`` into a list of case dicts."""
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if isinstance(raw, dict):
        # APPS-style {"inputs": [...], "outputs": [...], "fn_name": ...}
        if "inputs" in raw and "outputs" in raw:
            fn = raw.get("fn_name")
            cases = []
            for inp, out in zip(raw["inputs"], raw["outputs"]):
                if fn:
                    cases.append(
                        {"type": "call", "fn_name": fn, "input": inp, "output": out}
                    )
                else:
                    cases.append({"type": "stdin", "input": inp, "output": out})
            return cases
        raw = [raw]
    if not isinstance(raw, list):
        return []

    cases: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                continue
        if not isinstance(item, dict):
            continue
        t = str(item.get("testtype") or item.get("type") or "").lower()
        if not t:
            # Heuristic: presence of fn_name → call; else stdin
            t = "call" if item.get("fn_name") else "stdin"
        if t in ("functional", "call", "call_based", "function"):
            t = "call"
        else:
            t = "stdin"
        case: Dict[str, Any] = {
            "type": t,
            "input": item.get("input", item.get("inputs", "")),
            "output": item.get("output", item.get("outputs", "")),
        }
        if item.get("fn_name"):
            case["fn_name"] = item["fn_name"]
        cases.append(case)
    return cases


def _write_temp_py(source: str) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(source)
        f.flush()
        return f.name


def run_stdin_case(
    code: str,
    stdin_data: str,
    expected: str,
    *,
    timeout: int = 10,
) -> bool:
    """Run ``code`` as a script with ``stdin_data``; compare normalized stdout."""
    path: Optional[str] = None
    try:
        path = _write_temp_py(code)
        result = subprocess.run(
            [sys.executable, path],
            input=stdin_data if isinstance(stdin_data, str) else str(stdin_data),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return False
        return normalize_stdout(result.stdout) == normalize_stdout(str(expected))
    except (subprocess.TimeoutExpired, OSError, Exception):
        return False
    finally:
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass


def _parse_call_args(inp: Any) -> list:
    """Parse LCB functional input into a Python arg list.

    Official format: each line of the input string is a JSON-encoded argument.
    Also accepts a JSON list / already-parsed list.
    """
    if isinstance(inp, list):
        return inp
    if not isinstance(inp, str):
        return [inp]
    s = inp.strip()
    if not s:
        return []
    # Whole-string JSON list
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    args = []
    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            args.append(json.loads(line))
        except json.JSONDecodeError:
            args.append(line)
    return args


def run_call_case(
    code: str,
    *,
    fn_name: str,
    inp: Any,
    expected: Any,
    timeout: int = 10,
) -> bool:
    """Invoke ``fn_name(*args)`` (or ``Solution().fn_name``) and compare result."""
    args = _parse_call_args(inp)
    try:
        expected_val = (
            json.loads(expected) if isinstance(expected, str) else expected
        )
    except json.JSONDecodeError:
        expected_val = expected

    args_json = json.dumps(args)
    expected_json = json.dumps(expected_val)
    # Do not textwrap.dedent around user `code` — that breaks nested indentation.
    runner = (
        "import json\n"
        "import sys\n\n"
        f"{code}\n\n"
        f"args = json.loads({args_json!r})\n"
        f"expected = json.loads({expected_json!r})\n"
        f"fn_name = {fn_name!r}\n\n"
        "target = None\n"
        'if "Solution" in globals():\n'
        "    sol = Solution()\n"
        "    if hasattr(sol, fn_name):\n"
        "        target = getattr(sol, fn_name)\n"
        "if target is None:\n"
        "    target = globals().get(fn_name)\n"
        "if target is None:\n"
        "    sys.stderr.write(f'missing function {fn_name}\\n')\n"
        "    sys.exit(2)\n"
        "got = target(*args)\n"
        "if isinstance(got, tuple):\n"
        "    got = list(got)\n"
        "if got != expected:\n"
        "    sys.stderr.write(f'got={got!r} expected={expected!r}\\n')\n"
        "    sys.exit(1)\n"
        "sys.exit(0)\n"
    )
    path: Optional[str] = None
    try:
        path = _write_temp_py(runner)
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError, Exception):
        return False
    finally:
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass


def run_lcb_case(code: str, case: Dict[str, Any], *, timeout: int = 10) -> bool:
    ctype = str(case.get("type", "stdin")).lower()
    if ctype in ("call", "functional", "call_based", "function"):
        fn = case.get("fn_name") or "solve"
        return run_call_case(
            code,
            fn_name=str(fn),
            inp=case.get("input", case.get("args")),
            expected=case.get("output", case.get("expected")),
            timeout=timeout,
        )
    return run_stdin_case(
        code,
        str(case.get("input", "")),
        str(case.get("output", "")),
        timeout=timeout,
    )


def task_lcb_cases(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract structured LCB cases from a prepared task row."""
    cases = task.get("lcb_tests")
    if isinstance(cases, dict) and "cases" in cases:
        # Optional wrapper {"mode", "fn_name", "cases": [...]}
        base_fn = cases.get("fn_name")
        out = []
        for c in cases.get("cases") or []:
            c = dict(c)
            if base_fn and "fn_name" not in c:
                c["fn_name"] = base_fn
            if "type" not in c:
                c["type"] = cases.get("mode") or ("call" if base_fn else "stdin")
            out.append(c)
        return out
    if isinstance(cases, list):
        return [c for c in cases if isinstance(c, dict)]
    # Legacy: try parsing test_cases as public_test_cases JSON
    return parse_public_test_cases(task.get("test_cases"))


def pass_lcb(code: str, task: Dict[str, Any], *, timeout: int = 10) -> bool:
    """True if ``code`` passes every LCB case on the task."""
    if not code or not code.strip():
        return False
    cases = task_lcb_cases(task)
    if not cases:
        return False
    return all(run_lcb_case(code, c, timeout=timeout) for c in cases)


def fraction_lcb_ready(tasks: Sequence[Dict[str, Any]]) -> float:
    """Share of tasks that have at least one structured LCB case."""
    if not tasks:
        return 0.0
    ok = sum(1 for t in tasks if task_lcb_cases(t))
    return ok / len(tasks)
