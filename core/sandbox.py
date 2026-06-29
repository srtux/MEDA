import subprocess
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

class SandboxResult:
    def __init__(
        self,
        success: bool,
        stdout: str,
        stderr: str,
        returncode: int,
        metrics: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.metrics = metrics

class Sandbox:
    """Executes compiled CadQuery python code in an isolated subprocess to evaluate metrics and catch compile/logic errors."""
    def __init__(self, working_dir: str = "NewCADs"):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)

    def execute(self, code: str, timeout: float = 10.0) -> SandboxResult:
        # Append B-Rep topological extraction script to the end of the code
        extraction_code = """
# === TOPOLOGY EXTRACTION ===
import json
try:
    if 'model' in locals() or 'model' in globals():
        # model is usually a Workplane or Shape
        solid = model
        if hasattr(model, 'val'):
            val = model.val()
        else:
            val = model
            
        metrics = {
            "volume": float(val.Volume()),
            "area": float(val.Area()),
            "num_faces": int(len(model.faces().vals())) if hasattr(model, 'faces') else 0,
            "num_edges": int(len(model.edges().vals())) if hasattr(model, 'edges') else 0,
            "num_vertices": int(len(model.vertices().vals())) if hasattr(model, 'vertices') else 0,
            "center_of_mass": [float(val.Center().x), float(val.Center().y), float(val.Center().z)]
        }
        print(f"[METRICS_JSON] {json.dumps(metrics)}")
    else:
        print("[METRICS_ERROR] 'model' variable was not defined in the script.")
except Exception as e:
    print(f"[METRICS_ERROR] Failed to extract topology: {e}")
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
            
            success = (returncode == 0) and ("[COMPILE_SUCCESS]" in stdout)
            
            # Parse B-Rep metrics
            metrics = None
            for line in stdout.splitlines():
                if line.startswith("[METRICS_JSON]"):
                    try:
                        metrics_str = line.replace("[METRICS_JSON]", "").strip()
                        metrics = json.loads(metrics_str)
                    except Exception:
                        pass
                        
            return SandboxResult(
                success=success,
                stdout=stdout,
                stderr=stderr,
                returncode=returncode,
                metrics=metrics
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
