"""Kernel compilation in subprocess isolation.

Adapted from KernelBench ``load_custom_model_with_tempfile`` approach:
Triton's ``@triton.jit`` decorator does NOT work with ``exec()``, so we must write
the generated code to a temporary .py file and load it via ``importlib``.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class CompileResult:
    success: bool
    elapsed_s: float = 0.0
    stdout: str = ""
    stderr: str = ""
    error_type: str = ""  # "syntax", "import", "timeout", "unknown"
    tempfile_path: str = ""
    module_name: str = ""


def compile_in_subprocess(
    code: str,
    timeout_s: int = 120,
    cwd: Optional[str] = None,
) -> CompileResult:
    """Compile a Triton kernel in a crash-isolated subprocess.

    The subprocess writes the code to a tempfile, attempts to import it, and
    reports any errors via stdout/stderr.  The parent process never executes
    untrusted code directly.

    Returns a CompileResult that the caller can inspect for success/failure.
    """
    started = time.time()
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _COMPILE_CHECK_SCRIPT, code],
            cwd=cwd,
            timeout=timeout_s,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        return CompileResult(
            success=(completed.returncode == 0),
            elapsed_s=time.time() - started,
            stdout=completed.stdout,
            stderr=completed.stderr,
            error_type=_classify_compile_error(completed.stderr),
        )
    except subprocess.TimeoutExpired as exc:
        return CompileResult(
            success=False,
            elapsed_s=time.time() - started,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            error_type="timeout",
        )
    except Exception as exc:
        return CompileResult(
            success=False,
            elapsed_s=time.time() - started,
            stderr=str(exc),
            error_type=type(exc).__name__,
        )


def compile_in_process(code: str) -> CompileResult:
    """Compile a Triton kernel in-process via tempfile + importlib.

    Prefer ``compile_in_subprocess`` for untrusted code.  Use this only when
    you need to introspect the loaded module afterwards.
    """
    started = time.time()
    tmp: Optional[tempfile.NamedTemporaryFile] = None
    try:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="triton_kernel_"
        )
        tmp.write(code)
        tmp.flush()
        tmp_path = tmp.name

        spec = importlib.util.spec_from_file_location("_triton_kernel_mod", tmp_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_triton_kernel_mod"] = mod
        spec.loader.exec_module(mod)

        return CompileResult(
            success=True,
            elapsed_s=time.time() - started,
            tempfile_path=tmp_path,
            module_name="_triton_kernel_mod",
        )
    except SyntaxError as exc:
        return CompileResult(
            success=False,
            elapsed_s=time.time() - started,
            stderr=str(exc),
            error_type="syntax",
        )
    except Exception as exc:
        return CompileResult(
            success=False,
            elapsed_s=time.time() - started,
            stderr=str(exc),
            error_type="import",
        )
    finally:
        if tmp is not None:
            try:
                tmp.close()
            except Exception:
                pass


def cleanup_compiled_module(module_name: str, tempfile_path: str) -> None:
    """Remove tempfile and sys.modules entry after use."""
    sys.modules.pop(module_name, None)
    if tempfile_path and os.path.exists(tempfile_path):
        try:
            os.remove(tempfile_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Internal: the script that the subprocess runs
# ---------------------------------------------------------------------------

_COMPILE_CHECK_SCRIPT = r"""
import importlib, os, sys, tempfile, traceback

code = sys.argv[1]
tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix="triton_kernel_")
try:
    tmp.write(code)
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()

    spec = importlib.util.spec_from_file_location("_triton_check", tmp_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Report success + what we found
    print("COMPILE_OK", flush=True)
    attrs = [a for a in dir(mod) if not a.startswith("_")]
    print("attrs:", attrs, flush=True)
finally:
    try:
        os.remove(tmp.name)
    except OSError:
        pass
"""


def _classify_compile_error(stderr: str) -> str:
    """Heuristic to classify compilation errors from stderr output."""
    if not stderr:
        return ""
    lower = stderr.lower()
    if "syntaxerror" in lower or "indentationerror" in lower:
        return "syntax"
    if "modulenotfounderror" in lower or "importerror" in lower:
        return "import"
    if "timeout" in lower:
        return "timeout"
    return "runtime"
