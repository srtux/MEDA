import os
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from google import genai
from google.genai import types
from google.genai.errors import APIError
from dotenv import load_dotenv

# Load local libraries
from core.canvas import Canvas
from core.sandbox import Sandbox
from core.reward_engine import RewardEngine

load_dotenv()

# Global reference to prevent Pydantic serialization of genai.Client through bound instance methods
_active_core: Optional['ReasoningCADCore'] = None


# ==========================================
# GLOBAL MODULE-LEVEL TOOLS
# ==========================================

def add_parameter(name: str, value: float, description: str) -> str:
    """Add a named parametric variable (dimension, offset, diameter) to the design."""
    global _active_core
    if _active_core:
        _active_core.canvas.add_parameter(name, value, description)
        return f"Parameter '{name}' added successfully."
    return "Error: No active CAD session."

def set_parameter(name: str, value: float) -> str:
    """Update the value of an existing named parameter."""
    global _active_core
    if _active_core:
        if _active_core.canvas.set_parameter(name, value):
            return f"Parameter '{name}' updated to {value}."
        return f"Error: Parameter '{name}' does not exist."
    return "Error: No active CAD session."

def add_feature(code: str, description: str) -> str:
    """Append a new CadQuery CAD feature operation code block to the end of the timeline.
    Ensure the code assigns to 'model' variable.
    Example: model = cq.Workplane("XY").box(width, length, height)
    """
    global _active_core
    if _active_core:
        _active_core.canvas.add_feature(code, description)
        return f"Feature step appended to timeline."
    return "Error: No active CAD session."

def modify_feature(index: int, code: str, description: str) -> str:
    """Surgically modify or replace an existing feature step in the timeline."""
    global _active_core
    if _active_core:
        _active_core.log(f"[DEBUG_TOOL] modify_feature called with index={index}, features={[f.code for f in _active_core.canvas.features]}")
        if _active_core.canvas.modify_feature(index, code, description):
            return f"Feature step {index} modified successfully."
        return f"Error: Feature step index {index} out of range."
    return "Error: No active CAD session."

def remove_feature(index: int) -> str:
    """Surgically remove an unwanted feature step from the timeline."""
    global _active_core
    if _active_core:
        if _active_core.canvas.remove_feature(index):
            return f"Feature step {index} removed from timeline."
        return f"Error: Feature step index {index} out of range."
    return "Error: No active CAD session."

def run_cad_execution() -> str:
    """Compile and execute the current CAD script in the sandbox.
    Returns JSON containing success status, stdout/stderr, and topological B-Rep metrics.
    """
    global _active_core
    if _active_core:
        code = _active_core.canvas.to_python_code()
        res = _active_core.sandbox.execute(code)
        
        # Calculate reward internally to provide quick feedback
        reward, breakdown = RewardEngine.calculate_reward(res.success, res.metrics, _active_core.constraints)
        
        if not res.success:
            err_msg = "Compilation failure"
            if res.stderr:
                err_msg = f"Compile error: {res.stderr.strip()}"
            elif "[METRICS_ERROR]" in res.stdout:
                for line in res.stdout.split("\n"):
                    if line.startswith("[METRICS_ERROR]"):
                        err_msg = line
                        break
            breakdown["failed_constraints"].append(err_msg)

        # Multi-View Visual Critique step
        # If compilation is clean and B-Rep reward is 1.0 (or no constraints were violated), check visual correctness
        if res.success and reward == 1.0:
            png_path = _active_core.sandbox.working_dir / "001.png"
            try:
                import subprocess
                import sys
                script_path = Path("utils/capture_screenshot.py")
                # Ensure the STL file exists
                stl_file = _active_core.sandbox.working_dir / "001.stl"
                if stl_file.exists():
                    subprocess.run(
                        [sys.executable, str(script_path), str(stl_file), str(png_path)],
                        capture_output=True,
                        text=True
                    )
                    
                    if png_path.exists():
                        from PIL import Image
                        img = Image.open(png_path)
                        
                        visual_instruction = (
                            f"You are a CAD visual inspection expert. Compare the 4-view orthographic drawing collage of the 3D model with the design prompt: \"{_active_core.prompt}\".\n"
                            "Verify if the parts are correctly aligned and connected (e.g. wheels touch the ground/frame, head is attached to body).\n"
                            "Respond in JSON format with two keys:\n"
                            "- \"match\": true if it matches correctly without floating/distorted parts, false otherwise.\n"
                            "- \"critique\": a brief explanation of any misalignments, floating objects, or missing components.\n"
                        )
                        
                        resp = _active_core._safe_call(
                            _active_core.client.models.generate_content,
                            model="gemini-3.5-flash",  # Fast vision validation
                            contents=[img, visual_instruction],
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                temperature=0.0
                            )
                        )
                        
                        visual_res = json.loads(resp.text.strip())
                        _active_core.log(f"[LOG] Visual Critique - Match: {visual_res.get('match')}, Critique: {visual_res.get('critique')}")
                        
                        if not visual_res.get("match", True):
                            reward = 0.0  # Gate success until visuals are corrected
                            breakdown["failed_constraints"].append(f"Visual validation failure: {visual_res.get('critique')}")
            except Exception as e:
                _active_core.log(f"[WARNING] Visual validation step skipped: {e}")

        output = {
            "success": res.success,
            "stdout": res.stdout,
            "stderr": res.stderr,
            "metrics": res.metrics,
            "reward": reward,
            "failed_constraints": breakdown["failed_constraints"]
        }
        return json.dumps(output)
    return "Error: No active CAD session."


class ReasoningCADCore:
    """Autonomous reasoning agent loop driving CAD design using tool-use feedback."""
    def __init__(self, working_dir: str = "NewCADs"):
        self.canvas = Canvas()
        self.sandbox = Sandbox(working_dir)
        self.client = genai.Client()
        self.model_name = "gemini-3.5-flash"
        self.constraints: Dict[str, Any] = {}
        self.prompt = ""
        self.log_callback = None

    def log(self, message: str):
        print(message, flush=True)
        if self.log_callback:
            try:
                self.log_callback(message)
            except Exception:
                pass

    def _safe_call(self, func, *args, **kwargs):
        import time
        import random
        max_retries = 5
        backoff = 2.0
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except APIError as e:
                # Retry on 429 (RateLimit) and 503 (Unavailable)
                if getattr(e, 'code', None) in [429, 503] and attempt < max_retries - 1:
                    sleep_time = backoff + random.uniform(0, 1)
                    self.log(f"[WARNING] API returned {e.code}. Retrying in {sleep_time:.2f}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(sleep_time)
                    backoff *= 2.0
                else:
                    raise e
            except Exception as e:
                raise e

    def run_design_loop(
        self,
        prompt: str,
        constraints: Dict[str, Any],
        image_path: Optional[str] = None,
        max_iterations: int = 15
    ) -> Dict[str, Any]:
        """Runs the autonomous Chain-of-Thought tool execution loop until the design goal is achieved."""
        global _active_core
        _active_core = self
        
        self.log(f"\n[LOG] Initializing design loop for prompt: '{prompt}'")
        self.log(f"[LOG] Target Constraints: {constraints}")
        if image_path:
            self.log(f"[LOG] Image Path: {image_path}")
        
        # Reset canvas state
        self.canvas = Canvas()
        self.prompt = prompt
        # Extract/infer constraints from prompt using a simple model call
        inferred_constraints = {}
        try:
            system_extract = """
            You are a CAD parameter extraction assistant. Analyze the user design prompt and extract any implicit or explicit mathematical target constraints for the following B-Rep metrics (only if they are mentioned or can be calculated from the text):
            - "volume": target total volume in mm^3 (float)
            - "num_faces": exact number of faces (integer)
            - "num_edges": exact number of edges (integer)
            - "center_of_mass": [x, y, z] expected center of mass coordinates (list of floats)
            
            Return the output as a valid JSON object. Do not include markdown formatting or extra text.
            If no constraints can be calculated, return {}.
            """
            resp = self._safe_call(
                self.client.models.generate_content,
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_extract,
                    temperature=0.0
                )
            )
            text = resp.text.strip()
            # Clean markdown code blocks if any
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("\n", 1)[0].strip()
            if text.startswith("json"):
                text = text.split("json", 1)[1].strip()
            inferred_constraints = json.loads(text)
            self.log(f"[LOG] Inferred constraints from prompt: {inferred_constraints}")
        except Exception as e:
            self.log(f"[WARNING] Failed to infer constraints: {e}")

        # Merge constraints
        self.constraints = {**inferred_constraints, **constraints}
        
        # Standard system instructions
        system_instruction = """
You are an expert autonomous CAD engineering agent. Your objective is to design a 3D CAD model using CadQuery that perfectly matches the user design prompt and satisfies all target constraints.

You must build the design parametrically by calling the available tools in sequence:
1. Planning Phase (First Turn): Before calling any modeling tools, write a brief, step-by-step parametric modeling plan describing your design approach, coordinate layout, and any helper structures/functions (like line/strut generators or lofts) you plan to use.
2. Parameter Definition: Define all dimensions using the `add_parameter` tool.
3. Feature Timeline Construction: Add the feature code steps (starting with the base solid `model = ...`, then chaining subsequent modifications to `model = ...`) using the `add_feature` tool.
4. Compilation & Verification: Call the `run_cad_execution` tool to compile and verify the model's metrics.
5. Surgical Correction: If the execution returns a reward of 0.0, analyze the compile/constraint logs, use `modify_feature` or `set_parameter` to fix, and execute again. Repeat until the reward is 1.0.

Do NOT output a final text response declaring success until you have successfully executed `run_cad_execution` and verified that the reward is 1.0.

CRITICAL GEOMETRIC CONSTRAINTS:
*   **Revolve Axis Rules**: When calling `revolve(angle, axisStart, axisEnd)` on a 2D sketch workplane (e.g. `"XZ"` or `"YZ"`), the axis coordinate arguments are evaluated in the *local coordinate space* of that sketch plane. The axis of revolution MUST lie coplanar within the sketch plane (along local X `(1, 0, 0)` or local Y `(0, 1, 0)`). Never revolve around the local Z-axis `(0, 0, 1)` (perpendicular normal vector), as this is geometrically degenerate and crashes the OpenCascade kernel.
*   **Boolean Union Cleanliness**: When unioning secondary features (e.g. leaves, grips, pedals) to parent structures (e.g. stems, handlebars, cranks), attach them cleanly to their immediate parent and do not let them deeply or shallowly intersect other primary bodies (e.g. the main apple body). Deep intersections at shallow angles cause mesh singularities and distorted triangular face artifacts.
"""

        # Map global tool function handles with explicit schemas
        tools_list = [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="add_parameter",
                        description="Add a named parametric variable (dimension, offset, diameter) to the design.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "name": types.Schema(type="STRING", description="Name of the parameter"),
                                "value": types.Schema(type="NUMBER", description="Numeric value of the parameter"),
                                "description": types.Schema(type="STRING", description="Description of the parameter")
                            },
                            required=["name", "value", "description"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="set_parameter",
                        description="Update the value of an existing named parameter.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "name": types.Schema(type="STRING", description="Name of the parameter"),
                                "value": types.Schema(type="NUMBER", description="New numeric value")
                            },
                            required=["name", "value"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="add_feature",
                        description="Append a new CadQuery CAD feature operation code block to the end of the timeline.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "code": types.Schema(type="STRING", description="Code block updating 'model' variable"),
                                "description": types.Schema(type="STRING", description="Description of this feature step")
                            },
                            required=["code", "description"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="modify_feature",
                        description="Surgically modify or replace an existing feature step in the timeline.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "index": types.Schema(type="INTEGER", description="0-based index of the step to modify"),
                                "code": types.Schema(type="STRING", description="New code block updating 'model'"),
                                "description": types.Schema(type="STRING", description="New description of this feature step")
                            },
                            required=["index", "code", "description"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="remove_feature",
                        description="Surgically remove an unwanted feature step from the timeline.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "index": types.Schema(type="INTEGER", description="0-based index of the step to remove")
                            },
                            required=["index"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="run_cad_execution",
                        description="Compile and execute the current CAD script in the sandbox to verify it.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={}
                        )
                    )
                ]
            )
        ]
        
        chat = self._safe_call(
            self.client.chats.create,
            model=self.model_name,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=tools_list,
                temperature=0.0  # Force deterministic tool calling
            )
        )

        user_parts = [f"Design Goal: {prompt}\nTarget constraints: {json.dumps(self.constraints)}"]
        if image_path:
            img_path = Path(image_path)
            if img_path.exists():
                from PIL import Image
                try:
                    img = Image.open(img_path)
                    user_parts.append(img)
                except Exception as e:
                    self.log(f"[WARNING] Failed to load image context: {e}")
        
        iteration = 0
        current_reward = 0.0
        last_execution_result = {}
        
        # Initiate conversation
        response = self._safe_call(chat.send_message, user_parts)
        
        while iteration < max_iterations:
            iteration += 1
            self.log(f"\n--- CoT Iteration {iteration} of {max_iterations} ---")
            
            # Process function calls if the model returned them
            if response.function_calls:
                tool_responses = []
                for call in response.function_calls:
                    tool_name = call.name
                    args = call.args
                    self.log(f"[TOOL_CALL] Executing: {tool_name} with arguments: {args}")
                    
                    # Execute tool locally
                    if tool_name == "add_parameter":
                        res_str = add_parameter(args["name"], float(args["value"]), args["description"])
                    elif tool_name == "set_parameter":
                        res_str = set_parameter(args["name"], float(args["value"]))
                    elif tool_name == "add_feature":
                        res_str = add_feature(args["code"], args["description"])
                    elif tool_name == "modify_feature":
                        res_str = modify_feature(int(args["index"]), args["code"], args["description"])
                    elif tool_name == "remove_feature":
                        res_str = remove_feature(int(args["index"]))
                    elif tool_name == "run_cad_execution":
                        res_str = run_cad_execution()
                        # Parse reward state from execution output
                        try:
                            exec_data = json.loads(res_str)
                            current_reward = exec_data["reward"]
                            last_execution_result = exec_data
                        except Exception:
                            pass
                    else:
                        res_str = f"Error: Tool {tool_name} not recognized."
                        
                    self.log(f"[TOOL_RESPONSE] {res_str[:300]}...")
                    
                    # Store response for sending back to the model
                    tool_responses.append(
                        types.Part.from_function_response(
                            name=tool_name,
                            response={"result": res_str}
                        )
                    )
                
                # Send tool execution results back to the model to continue CoT loop
                response = self._safe_call(chat.send_message, tool_responses)
            else:
                # No more function calls, meaning the model finished reasoning
                self.log(f"[REASONING_RESPONSE] {response.text}")
                break
                
            if current_reward == 1.0:
                self.log("\n[LOG] Gated Reward target met (R = 1.0)! Ending design loop.")
                break

        # Auto-select suitable color hex based on prompt
        suggested_color = "#FF9900"
        if current_reward == 1.0:
            try:
                color_system = """
                You are a design color coordinator. Given a CAD model description prompt, return a suitable single hex color code (e.g. "#FF0800" for apple, "#8B5A2B" for wood box, "#708090" for steel tube) that represents the typical visual appearance of the object.
                Return the result as a raw JSON object containing only a "color" key. Do not include markdown formatting or extra text.
                """
                resp = self._safe_call(
                    self.client.models.generate_content,
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=color_system,
                        temperature=0.0,
                        response_mime_type="application/json"
                    )
                )
                color_data = json.loads(resp.text.strip())
                suggested_color = color_data.get("color", "#FF9900")
                self.log(f"[LOG] Auto-selected color {suggested_color} for prompt '{prompt}'")
            except Exception as e:
                self.log(f"[WARNING] Failed to auto-select color: {e}")

        final_code = self.canvas.to_python_code()
        
        # Save final files to NewCADs
        final_py_path = self.sandbox.working_dir / "001.py"
        with open(final_py_path, "w", encoding="utf-8") as f:
            f.write(final_code)
            
        # Compile and save diagnostic snapshot JSON
        diagnostic = {
            "prompt": prompt,
            "constraints": self.constraints,
            "iterations": iteration,
            "final_reward": current_reward,
            "final_code": final_code,
            "metrics": last_execution_result.get("metrics"),
            "failed_constraints": last_execution_result.get("failed_constraints", []),
            "suggested_color": suggested_color
        }
        diagnostic_path = self.sandbox.working_dir / "diagnostic.json"
        with open(diagnostic_path, "w", encoding="utf-8") as f:
            json.dump(diagnostic, f, indent=2)

        # Clean global pointer
        _active_core = None
            
        return {
            "success": current_reward == 1.0,
            "iterations": iteration,
            "final_reward": current_reward,
            "final_code": final_code,
            "metrics": last_execution_result.get("metrics"),
            "failed_constraints": last_execution_result.get("failed_constraints", []),
            "color": suggested_color
        }
