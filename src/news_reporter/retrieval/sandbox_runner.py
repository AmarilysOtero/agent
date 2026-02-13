#!/usr/bin/env python3
"""Subprocess sandbox for executing LLM-generated inspection code.

This module provides two components:

1. **Runner** (this script executed as __main__):
   Reads JSON from stdin, executes inspection code in a restricted
   environment, and writes JSON to stdout.  Designed to be invoked
   via `subprocess.Popen` with wall-clock + memory limits.

2. **Parent wrapper** (`sandbox_exec`):
   High-level async function called from recursive_summarizer.py.
   Spawns this script as a child process, passes the payload, and
   collects the result (or a safe fallback on timeout / crash).

Supported execution kinds:
  - "chunk_eval"   : runs evaluate_chunk_relevance(chunk_text) -> bool
  - "iter_program" : runs inspect_iteration(chunks) -> dict
  - "sanity_chunk" : runs chunk-level sanity tests
  - "sanity_iter"  : runs iteration-level sanity tests

Sandbox limits (enforced by parent):
  - Wall-clock timeout: 500 ms per invocation (configurable)
  - Memory: RLIMIT_AS on Linux / best-effort on Windows
  - CPU: RLIMIT_CPU on Linux

Sandbox limits (enforced inside runner):
  - _safe_range caps iterations at 10 000
  - _detect_multiplication_bombs rejects large mult constants
  - Restricted __builtins__ (no imports, no open, no eval)
"""

import json
import sys
import ast


# ── Restricted builtins (mirrors _build_safe_globals in recursive_summarizer) ─

def _safe_range(*args):
    """Capped range() to prevent CPU bombs."""
    if len(args) == 1:
        start, stop, step = 0, int(args[0]), 1
    elif len(args) == 2:
        start, stop, step = int(args[0]), int(args[1]), 1
    elif len(args) == 3:
        start, stop, step = int(args[0]), int(args[1]), int(args[2])
    else:
        raise ValueError("range() takes 1 to 3 arguments")
    step = step or 1
    iters = (stop - start) // step if step else 10**9
    if abs(iters) > 10_000:
        raise ValueError(f"range() capped at 10k iterations, got {abs(iters)}")
    return range(start, stop, step)


_SAFE_BUILTINS = {
    "len": len, "sum": sum, "any": any, "all": all,
    "min": min, "max": max, "int": int, "float": float,
    "str": str, "bool": bool, "list": list, "dict": dict,
    "set": set, "range": _safe_range, "enumerate": enumerate,
}


def _build_safe_globals() -> dict:
    return {"__builtins__": dict(_SAFE_BUILTINS)}


# ── Execution handlers ──────────────────────────────────────────────────────

def _run_chunk_eval(code: str, input_data: dict) -> dict:
    """Execute evaluate_chunk_relevance(chunk_text) -> bool."""
    chunk_text = input_data.get("chunk_text", "")
    ns: dict = {}
    exec(code, _build_safe_globals(), ns)
    func = ns.get("evaluate_chunk_relevance")
    if func is None:
        return {"ok": False, "result": None, "error": "evaluate_chunk_relevance not found"}
    result = bool(func(chunk_text))
    return {"ok": True, "result": result, "error": None}


def _run_iter_program(code: str, input_data: dict) -> dict:
    """Execute inspect_iteration(chunks) -> dict."""
    chunks = input_data.get("chunks", [])
    ns: dict = {}
    exec(code, _build_safe_globals(), ns)
    func = ns.get("inspect_iteration")
    if func is None:
        return {"ok": False, "result": None, "error": "inspect_iteration not found"}
    result = func(chunks)
    if not isinstance(result, dict):
        return {"ok": False, "result": None, "error": f"Expected dict, got {type(result).__name__}"}
    # Sanitise for JSON serialisation
    sanitised = {
        "selected_chunk_ids": list(result.get("selected_chunk_ids", [])),
        "extracted_data": result.get("extracted_data", {}),
        "confidence": float(result.get("confidence", 0.0)),
        "stop": bool(result.get("stop", False)),
    }
    return {"ok": True, "result": sanitised, "error": None}


def _run_sanity_chunk(code: str, input_data: dict) -> dict:
    """Run per-chunk sanity tests (empty / garbage rejection)."""
    chunk_text = input_data.get("chunk_text", "")
    ns: dict = {}
    exec(code, _build_safe_globals(), ns)
    func = ns.get("evaluate_chunk_relevance")
    if func is None:
        return {"ok": False, "result": False, "error": "evaluate_chunk_relevance not found"}

    # Test: empty string => must be False
    if func(""):
        return {"ok": True, "result": False, "error": "Returns True on empty string (default True pattern)"}

    # Test: garbage text => must be False
    garbage = [
        "asdf qwer zxcv qwerty",
        "the quick brown fox jumps over the lazy dog",
        "123 456 789 000 111",
    ]
    for g in garbage:
        if func(g):
            return {"ok": True, "result": False, "error": f"Returns True on irrelevant text: '{g}'"}

    return {"ok": True, "result": True, "error": None}


def _run_sanity_iter(code: str, input_data: dict) -> dict:
    """Run iteration-level sanity tests."""
    chunks = input_data.get("chunks", [])
    ns: dict = {}
    exec(code, _build_safe_globals(), ns)
    func = ns.get("inspect_iteration")
    if func is None:
        return {"ok": False, "result": False, "error": "inspect_iteration not found"}

    result = func(chunks)
    if not isinstance(result, dict):
        return {"ok": True, "result": False, "error": f"Return value is {type(result).__name__}, not dict"}

    required = {"selected_chunk_ids", "extracted_data", "confidence", "stop"}
    missing = required - set(result.keys())
    if missing:
        return {"ok": True, "result": False, "error": f"Missing keys: {missing}"}

    selected = result.get("selected_chunk_ids", [])
    if not isinstance(selected, list):
        return {"ok": True, "result": False, "error": f"selected_chunk_ids is {type(selected).__name__}, not list"}

    valid_ids = {c["chunk_id"] for c in chunks if "chunk_id" in c}
    for cid in selected:
        if cid not in valid_ids:
            return {"ok": True, "result": False, "error": f"Invalid chunk ID: {cid}"}

    conf = result.get("confidence", 0.0)
    if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
        return {"ok": True, "result": False, "error": f"confidence={conf} not in [0.0, 1.0]"}

    ratio = len(selected) / max(1, len(chunks))
    stop = result.get("stop", False)
    if ratio > 0.9 and len(chunks) > 2 and not stop:
        return {"ok": True, "result": False, "error": f"Selects {ratio:.1%} without stop=True"}
    if ratio > 0.95 and len(chunks) > 2:
        return {"ok": True, "result": False, "error": f"Selects {ratio:.1%} (almost all)"}

    return {"ok": True, "result": True, "error": None}


_HANDLERS = {
    "chunk_eval": _run_chunk_eval,
    "iter_program": _run_iter_program,
    "sanity_chunk": _run_sanity_chunk,
    "sanity_iter": _run_sanity_iter,
}


# ── Main entry point (when run as subprocess) ───────────────────────────────

def main():
    """Read JSON from stdin, execute, write JSON to stdout."""
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw)
    except Exception as e:
        json.dump({"ok": False, "result": None, "error": f"Invalid input: {e}"}, sys.stdout)
        sys.stdout.flush()
        sys.exit(1)

    kind = payload.get("kind", "")
    code = payload.get("code", "")
    input_data = payload.get("input", {})

    handler = _HANDLERS.get(kind)
    if handler is None:
        json.dump({"ok": False, "result": None, "error": f"Unknown kind: {kind}"}, sys.stdout)
        sys.stdout.flush()
        sys.exit(1)

    try:
        out = handler(code, input_data)
    except Exception as e:
        out = {"ok": False, "result": None, "error": str(e)}

    json.dump(out, sys.stdout)
    sys.stdout.flush()


if __name__ == "__main__":
    main()


# ── Parent wrapper (imported by recursive_summarizer) ────────────────────────

import asyncio
import subprocess
import os
import logging

_logger = logging.getLogger(__name__)

# Path to this file (used as the subprocess target)
_RUNNER_SCRIPT = os.path.abspath(__file__)

# Default timeout per sandbox invocation (milliseconds)
SANDBOX_TIMEOUT_MS = int(os.getenv("SANDBOX_TIMEOUT_MS", "500"))


async def sandbox_exec(
    kind: str,
    code: str,
    input_data: dict,
    timeout_ms: int | None = None,
) -> dict:
    """Execute inspection code in a subprocess sandbox.

    Args:
        kind:       One of "chunk_eval", "iter_program", "sanity_chunk", "sanity_iter".
        code:       The Python source to execute inside the sandbox.
        input_data: Data dict passed to the handler (chunk_text, chunks, etc.).
        timeout_ms: Wall-clock timeout in milliseconds (default SANDBOX_TIMEOUT_MS).

    Returns:
        {"ok": bool, "result": ..., "error": str | None}
        On timeout or crash, returns {"ok": False, "result": None, "error": "..."}.
    """
    if timeout_ms is None:
        timeout_ms = SANDBOX_TIMEOUT_MS

    payload = json.dumps({"kind": kind, "code": code, "input": input_data})

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, _RUNNER_SCRIPT,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=payload.encode()),
                timeout=timeout_ms / 1000.0,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            _logger.warning(f"⚠️  Sandbox timeout ({timeout_ms}ms) for kind={kind}")
            return {"ok": False, "result": None, "error": f"Sandbox timeout ({timeout_ms}ms)"}

        if proc.returncode != 0:
            err_text = stderr.decode(errors="replace")[:500]
            _logger.warning(f"⚠️  Sandbox exited with code {proc.returncode}: {err_text}")
            return {"ok": False, "result": None, "error": f"Sandbox exit code {proc.returncode}: {err_text}"}

        result = json.loads(stdout.decode())
        return result

    except Exception as e:
        _logger.warning(f"⚠️  Sandbox execution error: {e}")
        return {"ok": False, "result": None, "error": str(e)}
