"""
Code tools — Phase 5.

Write and run code, jailed to a workspace directory (~/.sylph/workspace). All
paths are confined there (no traversal out), and execution runs in a subprocess
with a hard timeout and that workspace as its cwd. Gated by the 'code' domain,
so it can be turned off entirely from the Dashboard Controls tab.

This is the most powerful capability, so it is deliberately narrow: write/read
files in the workspace, list it, and run a Python file or snippet. No shell, no
arbitrary path access.
"""

import asyncio
import logging
import os
import sys
import uuid

logger = logging.getLogger("sylph.tools.code")

WORKSPACE = os.path.expanduser("~/.sylph/workspace")
_RUN_TIMEOUT = float(os.environ.get("SYLPH_CODE_TIMEOUT", "30"))
_MAX_OUTPUT = 6000


def _ensure_workspace() -> str:
    os.makedirs(WORKSPACE, exist_ok=True)
    return WORKSPACE


def _safe_path(filename: str) -> str:
    """Resolve `filename` inside the workspace; reject anything that escapes it."""
    base = os.path.realpath(_ensure_workspace())
    p = os.path.realpath(os.path.join(base, filename))
    if p != base and not p.startswith(base + os.sep):
        raise ValueError("path is outside the workspace")
    return p


def write_code(filename: str, content: str) -> str:
    """Write a code/text file into the workspace."""
    filename = (filename or "").strip()
    if not filename:
        return "No filename given."
    try:
        path = _safe_path(filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content or "")
        logger.info("Wrote %d bytes to %s", len(content or ""), path)
        return f"Saved '{filename}' ({len(content or '')} bytes) to the workspace."
    except Exception as e:
        return f"Couldn't write '{filename}': {e}"


def read_code(filename: str) -> str:
    """Read a file from the workspace."""
    try:
        path = _safe_path((filename or "").strip())
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        return text[:_MAX_OUTPUT] + ("\n[truncated]" if len(text) > _MAX_OUTPUT else "")
    except FileNotFoundError:
        return f"No file '{filename}' in the workspace."
    except Exception as e:
        return f"Couldn't read '{filename}': {e}"


def list_workspace() -> list[str]:
    """List files in the workspace."""
    base = _ensure_workspace()
    out = []
    for root, _dirs, files in os.walk(base):
        for fn in files:
            rel = os.path.relpath(os.path.join(root, fn), base)
            out.append(rel)
    return out[:100]


async def run_python(filename: str = "", code: str = "") -> str:
    """
    Run a Python file from the workspace, or an inline `code` snippet, in a
    subprocess (cwd = workspace, timed out). Returns combined stdout+stderr.
    """
    _ensure_workspace()
    try:
        if filename.strip():
            target = _safe_path(filename.strip())
            if not os.path.exists(target):
                return f"No file '{filename}' to run."
            args = [sys.executable, target]
            label = filename
        elif code.strip():
            # Persist the snippet so tracebacks have a real filename.
            tmp = _safe_path(f"_snippet_{uuid.uuid4().hex[:6]}.py")
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(code)
            args = [sys.executable, tmp]
            label = "snippet"
        else:
            return "Nothing to run — give a filename or code."

        proc = await asyncio.create_subprocess_exec(
            *args, cwd=WORKSPACE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=_RUN_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            return f"'{label}' ran longer than {_RUN_TIMEOUT:.0f}s and was stopped."

        text = (out or b"").decode("utf-8", errors="replace").strip()
        if len(text) > _MAX_OUTPUT:
            text = text[:_MAX_OUTPUT] + "\n[output truncated]"
        rc = proc.returncode
        logger.info("Ran %s (exit %s, %d chars out)", label, rc, len(text))
        return f"(exit {rc})\n{text}" if text else f"'{label}' finished (exit {rc}) with no output."
    except Exception as e:
        return f"Couldn't run code: {e}"
