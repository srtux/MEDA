import os
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from google import genai
from google.genai import types
from google.genai.errors import APIError
from dotenv import load_dotenv

# Google ADK Imports
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.artifacts import InMemoryArtifactService
from google.adk.sessions import InMemorySessionService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService

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
                from utils.capture_screenshot import capture_orthographic_collage
                capture_orthographic_collage(str(_active_core.sandbox.working_dir / "001.stl"), str(png_path))
                
                # Invoke multimodal visual critic to evaluate match alignment
                visual_critique_system = """
                You are an expert mechanical engineering checker. Compare the rendered 3D CAD model views (collage of Isometric, Top, Front, Right orthographic projections) against the original text design prompt.
                Evaluate the model's structural features:
                1. Are all secondary parts (stems, leaves, handles, holes) correctly positioned relative to the main body?
                2. Are there any detached, overlapping, or intersecting surfaces causing distorted meshes?
                3. Does the design look physically accurate and completely aligned with the prompt?

                Return a JSON object containing two fields:
                - "match": boolean (true if visual alignment is correct and complete, false otherwise)
                - "critique": string (detailed critique describing any visual/positional errors, or explaining why it matches)
                Do not include markdown code block syntax.
                """
                from PIL import Image
                img = Image.open(png_path)
                
                critic_resp = _active_core._safe_call(
                    _active_core.client.models.generate_content,
                    model=_active_core.model_name,
                    contents=[img, f"Original prompt: {_active_core.prompt}"],
                    config=types.GenerateContentConfig(
                        system_instruction=visual_critique_system,
                        temperature=0.0,
                        response_mime_type="application/json"
                    )
                )
                critic_data = json.loads(critic_resp.text.strip())
                _active_core.log(f"[LOG] Visual Critique - Match: {critic_data.get('match')}, Critique: {critic_data.get('critique')}")
                
                if not critic_data.get("match", False):
                    # Visual fail sets reward back to 0.0 to force correction turn
                    reward = 0.0
                    breakdown["failed_constraints"].append(f"Visual validation failure: {critic_data.get('critique')}")
            except Exception as e:
                _active_core.log(f"[WARNING] Visual critique step encountered an error: {e}")
                reward = 0.0
                breakdown["failed_constraints"].append(f"Visual critique error: {str(e)}")

        output = {
            "success": res.success,
            "metrics": res.metrics,
            "reward": reward,
            "failed_constraints": breakdown["failed_constraints"]
        }
        return json.dumps(output)
    return "Error: No active CAD session."


class ReasoningCADCore:
    """Autonomous reasoning agent loop driving CAD design using tool-use feedback."""
    def __init__(self, working_dir: str = "NewCADs", model_name: str = "gemini-3.5-flash", api_key: Optional[str] = None):
        self.canvas = Canvas()
        self.sandbox = Sandbox(working_dir)
        self.model_name = model_name
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client()
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
        max_iterations: int = 100
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
        
        # ----------------------------------------------------
        # SYSTEM INSTRUCTIONS FOR MULTI-AGENT ADK ORCHESTRATION
        # ----------------------------------------------------
        
        orchestrator_instruction = """
You are the Lead CAD Architect and Coordinator (MEDAOrchestrator). Your objective is to design a 3D CAD model using CadQuery that perfectly matches the user design prompt and satisfies all target constraints.

You do not write code or compile geometry directly. Instead, you coordinate the workflow by delegating tasks to your specialized sub-agents:
1. **CADModelerAgent**: Delegate to this agent to define parametric variables, write feature code blocks, and surgically modify or remove feature timeline steps.
2. **VisualCriticAgent**: Delegate to this agent to execute the script in the sandbox, verify geometrical topology metrics (volume, faces, COM), evaluate visual alignment critiques on the rendered views, and verify the final reward.

**Your Delegation Workflow**:
- Step 1: Hand off to CADModelerAgent to construct the modeling plan, define parameters, and add the initial solid features.
- Step 2: Hand off to VisualCriticAgent to compile and visually verify the model's metrics.
- Step 3: Inspect the feedback from VisualCriticAgent. If the reward is 1.0, you are successful! End the loop.
- Step 4: If the reward is 0.0, analyze the failure reasons (compiler errors or visual critique feedback) and delegate back to CADModelerAgent with clear corrective instructions. Repeat this loop until VisualCriticAgent returns success (reward = 1.0).

Do NOT declare success in a text message to the user until VisualCriticAgent has run and confirmed that the reward is 1.0.
"""

        modeler_instruction = """
You are the CAD Modeler Specialist (CADModelerAgent). Your objective is to manage the parametric variables and feature timeline on the canvas.

You have access to the following tools:
- `add_parameter`: Define a new named variable (dimensions, counts, radii).
- `set_parameter`: Surgically update the value of an existing named parameter.
- `add_feature`: Append a new CadQuery CAD code block to the timeline (always modifying the 'model' solid variable).
- `modify_feature`: Surgically update or replace the code of an existing feature step.
- `remove_feature`: Surgically remove an unwanted feature step from the timeline.

**Your Workflow**:
- Define all necessary parameters first.
- Construct the solid features step-by-step.
- After making modification changes, return control to your coordinator so the visual critic can compile and verify them. Do not attempt to run verification yourself; you do not have sandbox execution tools.
"""

        critic_instruction = """
You are the Visual Critic and Verification Specialist (VisualCriticAgent). Your objective is to compile the CAD model, evaluate its topology metrics, and visually analyze its shape alignment.

You have access to the following tool:
- `run_cad_execution`: Compiles the canvas code, runs it inside the isolated sandbox, generates orthographic screenshot collages, triggers vision-based critiques, and calculates the reward score.

**Your Workflow**:
- Call `run_cad_execution` to execute the current timeline.
- Read the output JSON containing the success status, failed constraints, visual critiques, and reward value.
- Report these findings back to your coordinator. If there are failures, describe them clearly so the modeler can correct them.
"""

        # Setup specialized sub-agents
        modeler_agent = LlmAgent(
            name="CADModelerAgent",
            model=self.model_name,
            description="CAD modeler specialist that defines parameters and updates timeline features",
            instruction=modeler_instruction,
            tools=[add_parameter, set_parameter, add_feature, modify_feature, remove_feature]
        )

        critic_agent = LlmAgent(
            name="VisualCriticAgent",
            model=self.model_name,
            description="Verification specialist that compiles CAD models and runs visual checks",
            instruction=critic_instruction,
            tools=[run_cad_execution]
        )

        # Setup main orchestrator agent
        orchestrator_agent = LlmAgent(
            name="MEDAOrchestrator",
            model=self.model_name,
            description="Lead orchestrator agent that coordinates the design loop using handoffs",
            instruction=orchestrator_instruction,
            sub_agents=[modeler_agent, critic_agent]
        )

        # Setup ADK Runner with InMemory services
        runner = Runner(
            app_name=orchestrator_agent.name,
            agent=orchestrator_agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
            auto_create_session=True
        )

        # Build parts for initial message
        parts = [types.Part.from_text(text=f"Design Goal: {prompt}\nTarget constraints: {json.dumps(self.constraints)}")]
        if image_path:
            img_path = Path(image_path)
            if img_path.exists():
                import mimetypes
                mime_type, _ = mimetypes.guess_type(img_path)
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type or "image/png"))
                self.log(f"[LOG] Loaded image context: {image_path}")

        new_message = types.Content(role="user", parts=parts)
        session_id = f"session_{int(time.time())}"

        current_reward = 0.0
        last_execution_result = {}
        iteration = 0

        # Define async execution function to run the generator
        async def run_agent():
            nonlocal current_reward, last_execution_result, iteration
            async for event in runner.run_async(
                user_id="meda_user",
                session_id=session_id,
                new_message=new_message
            ):
                iteration += 1
                self.log(f"\n--- Turn {iteration} ---")
                
                # Stream thought texts
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            author_label = f"[{event.author}]" if event.author else "[Agent]"
                            self.log(f"{author_label} {part.text}")
                
                # Log agent transfers
                if event.actions and event.actions.transfer_to_agent:
                    self.log(f"\n[TRANSFER] Handoff from '{event.author}' to '{event.actions.transfer_to_agent}'")
                
                # Log function call triggers
                func_calls = event.get_function_calls()
                if func_calls:
                    for call in func_calls:
                        self.log(f"[{event.author} TOOL_CALL] Executing: {call.name} with arguments: {call.args}")
                
                # Log function responses and inspect run_cad_execution outputs
                func_responses = event.get_function_responses()
                if func_responses:
                    for resp in func_responses:
                        val = resp.response.get("result") if resp.response else ""
                        res_str = str(val) if val is not None else ""
                        self.log(f"[{event.author} TOOL_RESPONSE] {res_str[:300]}...")
                        if resp.name == "run_cad_execution":
                            try:
                                exec_data = json.loads(res_str)
                                current_reward = exec_data.get("reward", 0.0)
                                last_execution_result = exec_data
                            except Exception:
                                pass

        # Run the async loop synchronously
        import asyncio
        asyncio.run(run_agent())

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
