"""Execute LLM-generated CadQuery code in a hardened subprocess.

MEDA runs model-written Python to build geometry. The competitive roadmap
(``docs/text_to_cad_landscape_2026.md`` §8.5, tracked as audit item **M5**)
flags the bare ``subprocess`` as an RCE surface that should be sandboxed before
any hosted deployment. Full containerization (nsjail/firejail/Docker) isn't
always available, so this module adds the strongest portable hardening:

1. **AST allow-list validation** (default-deny): the generated code may only
   import a curated set of modeling modules and may not call dangerous builtins
   (``eval``/``exec``/``open``/``__import__``...) or touch sandbox-escape dunders
   (``__subclasses__``/``__globals__``...). Rejected code never runs.
2. **POSIX resource limits** via ``preexec_fn``: CPU-time and output-file-size
   caps as a runaway backstop (memory cap is opt-in to avoid breaking OCP's
   large virtual reservations — set ``MEDA_SANDBOX_MEMLIMIT_MB``).
3. **Secret scrubbing**: API keys are stripped from the child environment so a
   prompt-injected script cannot exfiltrate them.

On top of safety it now emits **CAD-Judge-style validity signals** (``is_valid``,
``num_solids``) alongside the topology metrics — a cheap compiler-as-verifier
check (roadmap §8.8) the reward/agent can use without a VLM call.
"""

import ast
import os
import re
import subprocess
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Output artifacts that must be regenerated every run. Stale copies from a
# previous iteration are deleted before execution so downstream consumers
# (reward scoring, screenshots) never read last turn's geometry (bug H1).
_STALE_ARTIFACTS = ("001.stl", "001.step", "001.png")

# --- AST allow-list ---------------------------------------------------------
# NOTE: AST allow-listing is *best-effort* hardening, not a true sandbox. Python
# has many string-indirection and frame-walk escape vectors; we close the known
# classes below, but the real defense for hosted/untrusted use is OS-level
# isolation (container/nsjail) plus _scrubbed_env() removing secrets entirely.
#
# Top-level module roots the generated code may import. Default-deny: anything
# else is rejected with an instructive message the agent can act on. ``operator``
# is deliberately NOT allowed — operator.attrgetter/methodcaller perform
# string-based attribute access that defeats the dunder checks below.
_ALLOWED_IMPORT_ROOTS = frozenset({
    "__future__", "cadquery", "math", "cmath", "json", "numpy", "random",
    "itertools", "functools", "copy", "typing", "dataclasses", "collections",
    "decimal", "statistics", "re", "OCP", "ocp_vscode", "ocp_tessellate",
})

# Builtins that enable code execution, I/O or escapes — never allowed in
# generated geometry code, whether *called* or merely *referenced* (a reference
# like ``_run = eval`` is just as dangerous as a direct ``eval(...)`` call).
_BANNED_CALLS = frozenset({
    "eval", "exec", "compile", "__import__", "open", "input", "exit", "quit",
    "breakpoint", "getattr", "setattr", "delattr", "globals", "locals", "vars",
    "memoryview", "help",
})

# Attribute names used for sandbox escapes via object introspection or
# frame/traceback/coroutine walks (e.g. ``e.__traceback__.tb_frame.f_builtins``
# reaches ``__import__`` with no banned import). Blocked wherever they appear.
_BANNED_ATTRS = frozenset({
    "__class__", "__bases__", "__base__", "__subclasses__", "__mro__",
    "__globals__", "__builtins__", "__dict__", "__getattribute__",
    "__reduce__", "__reduce_ex__", "__code__", "__import__", "__loader__",
    "__getattr__", "__setattr__", "__closure__",
    # Frame / traceback walk (reaches f_builtins['__import__'] with no import).
    "__traceback__", "tb_frame", "tb_next",
    "f_globals", "f_builtins", "f_locals", "f_back",
    # Generator / coroutine / async-gen frames (same reach).
    "gi_frame", "cr_frame", "ag_frame",
    # Bound-method / wrapper escapes.
    "__self__", "__func__", "__wrapped__",
    # Subclass / init hooks used in escape chains.
    "__class_getitem__", "__init_subclass__", "__subclasshook__", "__set_name__",
})

# str.format/format_map perform getattr/getitem on field specs whose dunder
# names live inside the literal, invisible to the Attribute/Name checks — e.g.
# ``"{0.__globals__[__builtins__]}".format(func)``. Heuristically flag literals
# with ``{...__`` field access.
_DUNDER_FMT_RE = re.compile(r"\{[^{}]*(?:\.__|\[__)")


def validate_code(code: str) -> Optional[str]:
    """Static AST check of *user* code. Returns a reason string if unsafe, else ``None``.

    Validate only the model-written code, before any trusted extraction snippet
    is appended. Best-effort: closes the known indirect-reference, ``operator``,
    frame-walk and str.format introspection escapes; not a substitute for OS
    isolation.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Let the real interpreter produce the canonical SyntaxError/traceback;
        # don't reject here so the existing compile-error feedback path is used.
        return None

    for node in ast.walk(tree):
        # Imports must resolve to an allow-listed root module.
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in _ALLOWED_IMPORT_ROOTS:
                    return (
                        f"disallowed import '{alias.name}'. Allowed roots: "
                        f"{', '.join(sorted(_ALLOWED_IMPORT_ROOTS))}."
                    )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root and root not in _ALLOWED_IMPORT_ROOTS:
                return (
                    f"disallowed import from '{node.module}'. Allowed roots: "
                    f"{', '.join(sorted(_ALLOWED_IMPORT_ROOTS))}."
                )
        # Banned builtin calls (by bare name).
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _BANNED_CALLS:
                return f"disallowed call to '{node.func.id}()'."
        # Banned dunder attribute access (escape vectors).
        elif isinstance(node, ast.Attribute):
            if node.attr in _BANNED_ATTRS:
                return f"disallowed attribute access '{node.attr}'."
        # Block name references to banned dunders OR to dangerous builtins.
        # The builtin check is restricted to Load context so legitimate
        # assignment targets (e.g. ``vars = [...]``) are not false-positives,
        # while every real bypass (``_run = eval``, ``[eval][0]``,
        # ``map(exec, ...)``, default-arg ``a=eval``) references the name in Load.
        elif isinstance(node, ast.Name):
            if node.id in _BANNED_ATTRS:
                return f"disallowed name '{node.id}'."
            if node.id in _BANNED_CALLS and isinstance(node.ctx, ast.Load):
                return f"disallowed reference to builtin '{node.id}'."
        # Block str.format-style dunder field access hidden inside literals.
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _DUNDER_FMT_RE.search(node.value):
                return "disallowed format-string dunder field access (str.format/format_map introspection)."

    return None


def _scrubbed_env() -> Dict[str, str]:
    """Child environment with secrets removed (prevents key exfiltration)."""
    secret_markers = ("API_KEY", "_KEY", "TOKEN", "SECRET", "PASSWORD", "OPENAI", "ANTHROPIC")
    env = {}
    for k, v in os.environ.items():
        ku = k.upper()
        if any(m in ku for m in secret_markers):
            continue
        env[k] = v
    return env


def _resource_limit_preexec(timeout: float):
    """Return a POSIX preexec_fn applying CPU/file-size (and opt-in memory) caps."""
    if os.name != "posix":
        return None

    def _apply():  # pragma: no cover - runs only in the child process
        try:
            import resource

            cpu = int(timeout) + 2
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
            # 256 MB output-file cap (generous for STL/STEP).
            fsize = 256 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
            mem_mb = os.environ.get("MEDA_SANDBOX_MEMLIMIT_MB")
            if mem_mb and mem_mb.isdigit():
                cap = int(mem_mb) * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        except Exception:
            # Never let limit-setting failure prevent execution.
            pass

    return _apply


class SandboxResult:
    def __init__(
        self,
        success: bool,
        stdout: str,
        stderr: str,
        returncode: int,
        metrics: Optional[Dict[str, Any]] = None,
        model_built: bool = False,
        exported: bool = False,
    ):
        self.success = success
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.metrics = metrics
        # Whether the solid was constructed (independent of export success).
        self.model_built = model_built
        # Whether STL/STEP export succeeded.
        self.exported = exported

class Sandbox:
    """Executes compiled CadQuery python code in a hardened isolated subprocess."""
    def __init__(self, working_dir: str = "NewCADs"):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)

    def execute(self, code: str, timeout: float = 10.0) -> SandboxResult:
        # Static safety gate: reject dangerous constructs before running (M5).
        reason = validate_code(code)
        if reason is not None:
            return SandboxResult(
                success=False,
                stdout="",
                stderr=f"[SECURITY_BLOCKED] Sandbox rejected the code: {reason}",
                returncode=-3,
            )

        # Remove stale output artifacts so a failed run cannot leave a previous
        # iteration's STL/PNG on disk to be mistaken for the current one (H1).
        for name in _STALE_ARTIFACTS:
            try:
                (self.working_dir / name).unlink(missing_ok=True)
            except OSError:
                pass

        # Append B-Rep topological extraction script to the end of the code.
        # We emit [MODEL_BUILT] as soon as the solid is constructed and metrics
        # are read from the resolved solid Shape (val), so build success is
        # measured independently of export success (H2) and topology counts come
        # from the solid rather than the Workplane selection stack (M3). We also
        # emit cheap CAD-Judge-style validity signals (is_valid, num_solids).
        extraction_code = """
# === TOPOLOGY EXTRACTION ===
import json as _json
try:
    if 'model' in locals() or 'model' in globals():
        _val = model.val() if hasattr(model, 'val') else model
        try:
            _num_solids = int(len(_val.Solids()))
        except Exception:
            _num_solids = None
        try:
            _is_valid = bool(_val.isValid())
        except Exception:
            _is_valid = None
        _metrics = {
            "volume": float(_val.Volume()),
            "area": float(_val.Area()),
            "num_faces": int(len(_val.Faces())),
            "num_edges": int(len(_val.Edges())),
            "num_vertices": int(len(_val.Vertices())),
            "center_of_mass": [float(_val.Center().x), float(_val.Center().y), float(_val.Center().z)],
            "num_solids": _num_solids,
            "is_valid": _is_valid
        }
        print("[MODEL_BUILT]")
        print(f"[METRICS_JSON] {_json.dumps(_metrics)}")
    else:
        print("[METRICS_ERROR] 'model' variable was not defined in the script.")
except Exception as _e:
    print(f"[METRICS_ERROR] Failed to extract topology: {_e}")
"""
        full_code = code + "\n" + extraction_code

        # Write to a temporary file in the working directory
        temp_file = self.working_dir / "sandbox_run.py"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(full_code)

        try:
            # Execute python interpreter in a separate hardened subprocess.
            result = subprocess.run(
                [sys.executable, temp_file.name],
                cwd=str(self.working_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=_scrubbed_env(),
                preexec_fn=_resource_limit_preexec(timeout),
            )

            stdout = result.stdout
            stderr = result.stderr
            returncode = result.returncode

            model_built = "[MODEL_BUILT]" in stdout
            exported = "[COMPILE_SUCCESS]" in stdout

            # A run is "successful" once the solid is built AND metrics were
            # extracted, even if STL/STEP export later failed. Export failure is
            # surfaced separately so a valid solid is not discarded (H2).
            metrics = None
            for line in stdout.splitlines():
                if line.startswith("[METRICS_JSON]"):
                    try:
                        metrics_str = line.replace("[METRICS_JSON]", "").strip()
                        metrics = json.loads(metrics_str)
                    except Exception:
                        pass

            success = (returncode == 0) and model_built and (metrics is not None)

            return SandboxResult(
                success=success,
                stdout=stdout,
                stderr=stderr,
                returncode=returncode,
                metrics=metrics,
                model_built=model_built,
                exported=exported,
            )

        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False,
                stdout="",
                stderr="Timeout: Code execution took longer than allowed.",
                returncode=-1
            )
        except Exception as e:
            return SandboxResult(
                success=False,
                stdout="",
                stderr=f"Sandbox execution invocation failed: {e}",
                returncode=-2
            )
        finally:
            # Clean up temp file
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
