import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import copy

class FeatureStep:
    """Represents a single parametric modeling step in the CAD feature tree."""
    def __init__(self, code: str, description: str = ""):
        self.code = code.strip()
        self.description = description

    def to_dict(self) -> Dict[str, str]:
        return {"code": self.code, "description": self.description}


class Canvas:
    """Centralized shared canvas object synchronizing the CAD parametric feature state and history."""
    def __init__(self):
        self.parameters: Dict[str, Dict[str, Any]] = {}
        self.features: List[FeatureStep] = []
        self.imports: List[str] = [
            "import cadquery as cq"
        ]
        self.history: List[Dict[str, Any]] = []

    def commit_to_history(self):
        """Saves current canvas state to history for undo/timeline support."""
        state = {
            "parameters": copy.deepcopy(self.parameters),
            "features": [f.to_dict() for f in self.features],
            "imports": list(self.imports)
        }
        self.history.append(state)

    def rollback(self) -> bool:
        """Rollbacks the canvas to the previous state. Returns True if successful."""
        if not self.history:
            return False
        # Load the previous state
        prev = self.history.pop()
        self.parameters = prev["parameters"]
        self.features = [FeatureStep(f["code"], f["description"]) for f in prev["features"]]
        self.imports = prev["imports"]
        return True

    def add_parameter(self, name: str, value: Any, description: str = ""):
        """Add or update a parametric variable."""
        self.commit_to_history()
        self.parameters[name] = {"value": value, "description": description}

    def set_parameter(self, name: str, value: Any) -> bool:
        """Update the value of an existing parameter."""
        if name in self.parameters:
            self.commit_to_history()
            self.parameters[name]["value"] = value
            return True
        return False

    def add_feature(self, code: str, description: str = ""):
        """Append a feature operation step to the end of the timeline."""
        self.commit_to_history()
        self.features.append(FeatureStep(code, description))

    def insert_feature(self, index: int, code: str, description: str = ""):
        """Insert a feature operation step at a specific timeline position."""
        self.commit_to_history()
        self.features.insert(index, FeatureStep(code, description))

    def remove_feature(self, index: int) -> bool:
        """Remove a feature operation step from the timeline."""
        if 0 <= index < len(self.features):
            self.commit_to_history()
            self.features.pop(index)
            return True
        return False

    def modify_feature(self, index: int, code: str, description: str = "") -> bool:
        """Modify an existing feature operation step code and description."""
        if 0 <= index < len(self.features):
            self.commit_to_history()
            self.features[index] = FeatureStep(code, description)
            return True
        return False

    def to_python_code(self) -> str:
        """Compiles the parameter tree and CAD feature sequence into an executable Python script."""
        code_lines = []
        
        # 1. Imports
        code_lines.append("# === IMPORTS ===")
        for imp in self.imports:
            code_lines.append(imp)
        code_lines.append("")

        # 2. Parameters
        code_lines.append("# === PARAMETERS ===")
        for name, data in self.parameters.items():
            desc = f"  # {data['description']}" if data['description'] else ""
            val = data['value']
            # Format strings with quotes
            if isinstance(val, str):
                code_lines.append(f"{name} = '{val}'{desc}")
            else:
                code_lines.append(f"{name} = {val}{desc}")
        code_lines.append("")

        # 3. Model Generation
        code_lines.append("# === MODEL GENERATION ===")
        # Start constructing the box/shape sequence
        # We assume the first feature step initializes the main shape and binds it to 'model'
        # Subsequent steps modify 'model' or chain operations.
        # Example format:
        # Step 0: model = cq.Workplane("XY").box(width, length, height)
        # Step 1: model = model.faces(">Z").workplane().hole(hole_diameter)
        for i, step in enumerate(self.features):
            code_lines.append(f"# Step {i}: {step.description}")
            code_lines.append(step.code)
            code_lines.append("")

        # 4. Exports (standardized)
        code_lines.append("# === EXPORTS ===")
        code_lines.append("try:")
        code_lines.append("    cq.exporters.export(model, '001.stl')")
        code_lines.append("    cq.exporters.export(model, '001.step')")
        code_lines.append("    print('[COMPILE_SUCCESS]')")
        code_lines.append("except Exception as e:")
        code_lines.append("    print(f'[EXPORT_ERROR] {e}')")
        
        return "\n".join(code_lines)
