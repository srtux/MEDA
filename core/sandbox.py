import subprocess
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Output artifacts that must be regenerated every run. Stale copies from a
# previous iteration are deleted before execution so downstream consumers
# (reward scoring, screenshots) never read last turn's geometry (bug H1).
_STALE_ARTIFACTS = ("001.stl", "001.step", "001.png")


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
    """Executes compiled CadQuery python code in an isolated subprocess to evaluate metrics and catch compile/logic errors."""
    def __init__(self, working_dir: str = "NewCADs"):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)

    def execute(self, code: str, timeout: float = 10.0) -> SandboxResult:
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
        # from the solid rather than the Workplane selection stack (M3).
        extraction_code = """
# === TOPOLOGY EXTRACTION ===
import json as _json
try:
    if 'model' in locals() or 'model' in globals():
        _val = model.val() if hasattr(model, 'val') else model
        _metrics = {
            "volume": float(_val.Volume()),
            "area": float(_val.Area()),
            "num_faces": int(len(_val.Faces())),
            "num_edges": int(len(_val.Edges())),
            "num_vertices": int(len(_val.Vertices())),
            "center_of_mass": [float(_val.Center().x), float(_val.Center().y), float(_val.Center().z)]
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
            # Execute python interpreter in a separate subprocess
            result = subprocess.run(
                [sys.executable, temp_file.name],
                cwd=str(self.working_dir),
                capture_output=True,
                text=True,
                timeout=timeout
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
