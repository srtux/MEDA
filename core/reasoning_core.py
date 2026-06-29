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
from core.learning_store import LearningStore
from core.lesson_signature import to_signature
from core.memory import CADMemoryStore
from core.cad_knowledge import KNOWLEDGE_BASE
from core.geometry_introspection import INTROSPECTION_SNIPPET, parse_report, format_report
from core import candidate_search

load_dotenv()

# Global reference to prevent Pydantic serialization of genai.Client through bound instance methods
_active_core: Optional['ReasoningCADCore'] = None

# The module-level tools below operate on `_active_core`, a single global. Runs
# are therefore serialized with this lock so two concurrent Streamlit sessions
# cannot clobber each other's canvas/sandbox (bug H4).
import threading
_run_lock = threading.Lock()


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

def lookup_cadquery_api(query: str) -> str:
    """Look up the correct CadQuery API for an operation before writing code.

    Returns signatures, gotchas and worked examples for the methods most
    relevant to ``query`` (e.g. "drill a counterbored hole", "fillet vertical
    edges", "hollow shell"). Use this whenever unsure which method/selector to
    call, instead of guessing a method name.
    """
    global _active_core
    embed_fn = None
    if _active_core is not None and getattr(_active_core, "store", None) is not None:
        embed_fn = _active_core.store._embed
    try:
        return KNOWLEDGE_BASE.lookup(query, k=4) if embed_fn is None else \
            (KNOWLEDGE_BASE.retrieve_block(query, k=4, embed_fn=embed_fn) or
             KNOWLEDGE_BASE.lookup(query, k=4))
    except Exception:
        return KNOWLEDGE_BASE.lookup(query, k=4)


def inspect_current_model() -> str:
    """Measure the current solid's real faces and edges to ground selectors.

    Compiles the current timeline and reports each face's orientation/center and
    each edge's orientation/length, plus ready-to-use selector hints (e.g.
    "4x vertical (|Z)"). Call this BEFORE applying a fillet, chamfer, shell or
    face-targeted hole so the selector matches geometry that actually exists.
    """
    global _active_core
    if not _active_core:
        return "Error: No active CAD session."
    core = _active_core
    if not core.canvas.features:
        return "No features on the timeline yet; add a base solid before inspecting geometry."
    code = core.canvas.to_python_code() + "\n" + INTROSPECTION_SNIPPET
    res = core.sandbox.execute(code)
    report = parse_report(res.stdout)
    if report is None:
        err = (res.stderr or "").strip()[:200]
        return f"Could not inspect geometry: the model did not build. {err}"
    return format_report(report)


def run_cad_execution() -> str:
    """Compile and execute the current CAD script in the sandbox.

    Renders the model on every successful compile (so the agent sees its
    progress each turn), recalls relevant lessons for any failures, records new
    lessons when a past failure is resolved, and returns a JSON feedback blob.
    """
    global _active_core
    if not _active_core:
        return "Error: No active CAD session."

    core = _active_core
    code = core.canvas.to_python_code()
    res = core.sandbox.execute(code)

    if res.success:
        core.successful_compiles_count += 1
        iter_prefix = f"001_iter_{core.successful_compiles_count}"
        import shutil
        try:
            shutil.copy2(core.sandbox.working_dir / "001.stl", core.sandbox.working_dir / f"{iter_prefix}.stl")
            with open(core.sandbox.working_dir / f"{iter_prefix}.py", "w", encoding="utf-8") as f:
                f.write(code)
        except Exception as e:
            core.log(f"[WARNING] Failed to save intermediate iteration files: {e}")

    # Only score against an on-disk STL that the current run actually produced
    # (the sandbox deletes stale artifacts up front, so a missing file means
    # this turn did not export one).
    stl_file = core.sandbox.working_dir / "001.stl"
    generated_stl = str(stl_file) if (res.success and stl_file.exists()) else None
    reward, breakdown = RewardEngine.calculate_reward(
        res.success, res.metrics, core.constraints, generated_stl=generated_stl
    )

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
    elif not res.exported:
        # Solid built but STL/STEP export failed: surface as a soft warning so
        # the modeler can fix manifold issues, without discarding valid metrics.
        breakdown["failed_constraints"].append(
            "Export warning: solid built but STL/STEP export failed (possible non-manifold geometry)."
        )

    # Render the current model on every successful compile (not just reward==1.0)
    # so the agent gets visual feedback each iteration.
    render_path = None
    render_bytes = None
    if res.success:
        render_path, render_bytes = core.render_current_model(code)

    # Multi-View Visual Critique: only escalate to the (costly) vision model
    # once the B-Rep reward is already 1.0, reusing the render captured above.
    if res.success and reward == 1.0:
        try:
            if render_bytes is None:
                raise RuntimeError("no render available for visual critique")
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
            import io
            from PIL import Image
            img = Image.open(io.BytesIO(render_bytes))

            critic_resp = core._safe_call(
                core.client.models.generate_content,
                model=core.model_name,
                contents=[img, f"Original prompt: {core.prompt}"],
                config=types.GenerateContentConfig(
                    system_instruction=visual_critique_system,
                    temperature=0.0,
                    response_mime_type="application/json"
                )
            )
            critic_data = json.loads(critic_resp.text.strip())
            core.log(f"[LOG] Visual Critique - Match: {critic_data.get('match')}, Critique: {critic_data.get('critique')}")

            if not critic_data.get("match", False):
                reward = 0.0
                breakdown["failed_constraints"].append(f"Visual validation failure: {critic_data.get('critique')}")
        except Exception as e:
            core.log(f"[WARNING] Visual critique step encountered an error: {e}")
            reward = 0.0
            breakdown["failed_constraints"].append(f"Visual critique error: {str(e)}")

    # --- Self-learning: recall relevant past lessons, learn from this turn ---
    recalled = core.recall_lessons(breakdown["failed_constraints"])
    core.learn_from_transition(breakdown["failed_constraints"], reward, code)

    output = {
        "success": res.success,
        "metrics": res.metrics,
        "reward": reward,
        "geom_score": breakdown.get("geom_score"),
        "distances": breakdown.get("distances"),
        "failed_constraints": breakdown["failed_constraints"],
        "recalled_lessons": recalled,
        "render_path": render_path,
    }
    return json.dumps(output)


from functools import cached_property
from google.adk.models import Gemini

class RetryingGemini(Gemini):
    _api_key: Optional[str] = None

    @cached_property
    def api_client(self) -> genai.Client:
        # Configure robust HTTP retry options for the GenAI client
        retry_opts = types.HttpRetryOptions(
            attempts=8,
            initial_delay=3.0,
            max_delay=60.0,
            http_status_codes=[408, 429, 500, 502, 503, 504]
        )
        http_opts = types.HttpOptions(
            retry_options=retry_opts
        )
        if self._api_key:
            return genai.Client(api_key=self._api_key, http_options=http_opts)
        return genai.Client(http_options=http_opts)


class ReasoningCADCore:
    """Autonomous reasoning agent loop driving CAD design using tool-use feedback."""
    
    ORCHESTRATOR_INSTRUCTION = """
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

    MODELER_INSTRUCTION = """
You are the CAD Modeler Specialist (CADModelerAgent). Your objective is to manage the parametric variables and feature timeline on the canvas.

You have access to the following tools:
- `add_parameter`: Define a new named variable (dimensions, counts, radii).
- `set_parameter`: Surgically update the value of an existing named parameter.
- `add_feature`: Append a new CadQuery CAD code block to the timeline (always modifying the 'model' solid variable).
- `modify_feature`: Surgically update or replace the code of an existing feature step.
- `remove_feature`: Surgically remove an unwanted feature step from the timeline.
- `lookup_cadquery_api`: Look up the correct CadQuery method signature, gotchas and an example for an operation (e.g. "counterbored hole", "fillet vertical edges", "hollow shell"). Call this BEFORE writing an operation you are unsure about — never guess a method name.
- `inspect_current_model`: Measure the real faces and edges of the current solid (orientations, lengths, selector hints). Call this BEFORE a fillet, chamfer, shell, or face-targeted hole so your selector (e.g. `'|Z'`, `'>Z'`) matches edges/faces that actually exist and your fillet/chamfer radius stays under the safe ceiling it reports.

**Your Workflow**:
- Define all necessary parameters first.
- Construct the solid features step-by-step. Prefer the exact methods/selectors returned by `lookup_cadquery_api`.
- Before any advanced feature (fillet/chamfer/shell/face hole), call `inspect_current_model` and choose selectors grounded in the measured geometry.
- After making modification changes, return control to your coordinator so the visual critic can compile and verify them. Do not attempt to run verification yourself; you do not have sandbox execution tools.
"""

    CRITIC_INSTRUCTION = """
You are the Visual Critic and Verification Specialist (VisualCriticAgent). Your objective is to compile the CAD model, evaluate its topology metrics, and visually analyze its shape alignment.

You have access to the following tool:
- `run_cad_execution`: Compiles the canvas code, runs it inside the isolated sandbox, generates orthographic screenshot collages, triggers vision-based critiques, and calculates the reward score.

**Your Workflow**:
- Call `run_cad_execution` to execute the current timeline.
- Read the output JSON containing the success status, failed constraints, visual critiques, and reward value.
- Report these findings back to your coordinator. If there are failures, describe them clearly so the modeler can correct them.
"""

    SINGLE_SHOT_SYSTEM = """
            You are an expert parametric CAD script writer. Your task is to write a complete, self-contained Python script using the CadQuery library to design the requested CAD model.

            Follow these rules:
            1. Always import cadquery as cq: `import cadquery as cq`
            2. Declare all parametric variables (dimensions, offsets, radii, heights) clearly at the top of the script.
            3. Build the CAD geometry step-by-step.
            4. Assign the final solid shape or cq.Assembly object to the variable named `model`.
            5. Export the model at the end:
               try:
                   cq.exporters.export(model, '001.stl')
                   cq.exporters.export(model, '001.step')
                   print('[COMPILE_SUCCESS]')
               except Exception as e:
                   print(f'[EXPORT_ERROR] {e}')
            6. Return ONLY the raw Python code. Do not wrap it in markdown code blocks or triple backticks.
            """

    def __init__(self, working_dir: str = "NewCADs", model_name: str = "gemini-3.5-flash", api_key: Optional[str] = None):
        self.canvas = Canvas()
        self.sandbox = Sandbox(working_dir)
        self.model_name = model_name
        self._api_key_override = api_key
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client()
        self.constraints: Dict[str, Any] = {}
        self.prompt = ""
        self.log_callback = None
        self.event_callback = None
        self.successful_compiles_count = 0

        # Live per-iteration render callback (set by the UI): (iter_n, png_path).
        self.render_callback = None

        # Durable cross-run learning store (lessons + reusable skills).
        try:
            self.store: Optional[LearningStore] = LearningStore(genai_client=self.client)
        except Exception as e:
            self.store = None
            print(f"[WARNING] Learning store unavailable: {e}", flush=True)

        # Lightweight cross-run trajectory memory (compact per-run lessons).
        # Previously referenced as ``self.memory_store`` but never instantiated,
        # so every record_run() call silently failed — now wired up.
        try:
            self.trajectory_memory: Optional[CADMemoryStore] = CADMemoryStore()
        except Exception as e:
            self.trajectory_memory = None
            print(f"[WARNING] Trajectory memory unavailable: {e}", flush=True)

        # Per-run render bookkeeping.
        self.render_iter = 0
        self.last_geom_hash: Optional[str] = None
        self.last_png_path: Optional[str] = None
        self.last_png_bytes: Optional[bytes] = None

        # Per-run learning bookkeeping.
        self._surfaced_lessons: Dict[str, str] = {}   # signature -> lesson id
        self._prev_failed_sigs: set = set()
        self._prev_detail: Dict[str, str] = {}
        self._prev_reward = 0.0

    def _reset_run_state(self):
        self.render_iter = 0
        self.last_geom_hash = None
        self.last_png_path = None
        self.last_png_bytes = None
        self._surfaced_lessons = {}
        self._prev_failed_sigs = set()
        self._prev_detail = {}
        self._prev_reward = 0.0

    # ---------------------------------------------------------- rendering
    def render_current_model(self, compiled_code: str):
        """Render the freshly-compiled model to ``iter_{n}.png``.

        Skips re-rendering when the compiled code is unchanged (geometry hash),
        promotes the latest render to ``001.png`` for the final UI, and pushes
        it to the live render callback. Returns ``(png_path, png_bytes)`` or
        ``(None, None)`` when no STL is available or rendering fails.
        """
        import hashlib
        import shutil
        from pathlib import Path as _Path

        geom_hash = hashlib.sha1(compiled_code.encode("utf-8")).hexdigest()
        if (geom_hash == self.last_geom_hash and self.last_png_path
                and _Path(self.last_png_path).exists()):
            return self.last_png_path, self.last_png_bytes

        wd = self.sandbox.working_dir
        stl_path = wd / "001.stl"
        if not stl_path.exists():
            return None, None

        self.render_iter += 1
        n = self.render_iter
        png_path = wd / f"iter_{n}.png"
        try:
            # Reuses matplotlib headless screenshot capture from utils
            from utils.capture_screenshot import capture_orthographic_collage
            png_bytes = capture_orthographic_collage(str(stl_path), str(png_path))
        except Exception as e:
            self.log(f"[WARNING] Render failed: {e}")
            return None, None
        if png_bytes is None:
            return None, None

        try:
            shutil.copy(png_path, wd / "001.png")
        except Exception:
            pass

        self.last_geom_hash = geom_hash
        self.last_png_path = str(png_path)
        self.last_png_bytes = png_bytes
        self.log(f"[LOG] Rendered iteration {n} -> {png_path.name}")
        if self.render_callback:
            try:
                self.render_callback(n, str(png_path))
            except Exception:
                pass
        return str(png_path), png_bytes


    def _save_success_artifacts(self, code: str, context: str = "design loop") -> Optional[str]:
        """Persist the successful run as an iteration snapshot and render it.

        The sandbox writes canonical ``001.stl``/``001.step`` artifacts. This
        helper mirrors them into the iteration history, writes the matching
        Python snapshot, and promotes a fresh render to ``001.png`` through
        :meth:`render_current_model`. It keeps fast mode, multi-candidate mode,
        and the agentic loop on the same UI artifact contract.
        """
        import shutil

        stl_path = self.sandbox.working_dir / "001.stl"
        if not stl_path.exists():
            self.log(f"[WARNING] Cannot save {context} artifacts: 001.stl is missing.")
            return None

        self.successful_compiles_count += 1
        iter_prefix = f"001_iter_{self.successful_compiles_count}"
        try:
            shutil.copy2(stl_path, self.sandbox.working_dir / f"{iter_prefix}.stl")
            step_path = self.sandbox.working_dir / "001.step"
            if step_path.exists():
                shutil.copy2(step_path, self.sandbox.working_dir / f"{iter_prefix}.step")
            with open(self.sandbox.working_dir / f"{iter_prefix}.py", "w", encoding="utf-8") as f:
                f.write(code)
        except Exception as e:
            self.log(f"[WARNING] Failed to save {context} iteration files: {e}")

        self.render_current_model(code)
        return iter_prefix

    # ---------------------------------------------------------- learning
    def build_memory_preamble(self, prompt: str) -> str:
        """Build the modeler preamble: retrieved CadQuery API docs + skills + lessons.

        The API reference block (RAG grounding, roadmap §8.1) is always included
        because it is always relevant to CadQuery codegen and is the cheapest
        defense against hallucinated methods. Skills/lessons/trajectory tips are
        added when the learning store has relevant entries.
        """
        lines: List[str] = []

        # 1. CadQuery API RAG grounding (always-on; works offline via keywords).
        try:
            embed_fn = self.store._embed if self.store else None
            api_block = KNOWLEDGE_BASE.retrieve_block(prompt, k=6, embed_fn=embed_fn)
            if api_block:
                lines.append(api_block)
        except Exception as e:
            self.log(f"[WARNING] API doc retrieval failed: {e}")

        # 2. Reusable skills + lessons from the durable learning store.
        if self.store:
            try:
                skills = self.store.retrieve_skills(prompt, k=4)
                lessons = self.store.retrieve_lessons(prompt, k=3)
            except Exception as e:
                self.log(f"[WARNING] Memory retrieval failed: {e}")
                skills, lessons = [], []
            if skills:
                lines.append("\n## Reusable CadQuery skills (parameterized snippets that worked before):")
                for s in skills:
                    lines.append(f"- {s['name']} {s['signature']}: {s['goal_description']}")
                    lines.append(f"    {s['code_template']}")
            if lessons:
                lines.append("\n## Lessons from past failures (do NOT repeat these mistakes):")
                for l in lessons:
                    lines.append(f"- [{l['error_signature']}] {l['root_cause']} FIX: {l['corrective_fix']}")

        # 3. Compact trajectory tips from prior similar runs.
        if self.trajectory_memory:
            try:
                mems = self.trajectory_memory.retrieve(prompt, limit=2)
            except Exception:
                mems = []
            if mems:
                lines.append("\n## Tips from prior similar design runs:")
                for m in mems:
                    lines.append(f"- ({m.outcome}) {m.tip}")

        return "\n".join(lines).strip()

    def recall_lessons(self, failed_constraints):
        """For each current failure, recall the most similar past lessons."""
        if not self.store or not failed_constraints:
            return []
        recalled = []
        for fc in failed_constraints:
            sig = to_signature(fc)
            try:
                hits = self.store.retrieve_lessons(f"{sig}\n{self.prompt}", k=2, signature=sig)
            except Exception:
                hits = []
            for h in hits:
                self._surfaced_lessons[sig] = h["id"]
            recalled.extend(hits)
        # Dedup by id, cap size.
        seen, out = set(), []
        for r in recalled:
            if r["id"] not in seen:
                seen.add(r["id"])
                out.append(r)
        return out[:4]

    def learn_from_transition(self, current_failed, reward, code):
        """Update the lesson store based on what changed since the last turn."""
        if not self.store:
            return
        curr_sigs = {to_signature(fc) for fc in current_failed}
        resolved = self._prev_failed_sigs - curr_sigs

        # A prior failure is now resolved: reward the surfaced lesson that helped
        # and persist the working fix as the lesson's corrective_fix. (The store
        # exposes record_lesson + feedback; the previously-called
        # record_lesson_resolution/record_failure methods never existed, so this
        # whole loop silently no-op'd behind the try/except.)
        for sig in resolved:
            self.log(f"[LOG] Lesson learned! Prior failure [{sig}] was successfully resolved.")
            lesson_id = self._surfaced_lessons.get(sig)
            try:
                if lesson_id:
                    self.store.feedback(lesson_id, helped=True)
                detail = self._prev_detail.get(sig, "")
                self.store.record_lesson(
                    error_signature=sig,
                    error_detail=detail,
                    root_cause=detail,
                    corrective_fix=code,
                    prompt_context=self.prompt,
                )
            except Exception as e:
                self.log(f"[WARNING] Failed to record lesson resolution: {e}")

        # Negative feedback: record a failure-context lesson (no fix yet) for any
        # persistent/new failure when the reward did not improve. A later
        # resolution merges into the same signature and fills in corrective_fix.
        if reward <= self._prev_reward and curr_sigs:
            for sig in curr_sigs:
                desc = self._prev_detail.get(sig, "")
                if not desc:
                    # Find detail string in current failed list
                    for fc in current_failed:
                        if to_signature(fc) == sig:
                            desc = fc
                            break
                try:
                    self.store.record_lesson(
                        error_signature=sig,
                        error_detail=desc,
                        root_cause=desc,
                        corrective_fix="",
                        prompt_context=self.prompt,
                    )
                except Exception as e:
                    self.log(f"[WARNING] Failed to record failure lesson: {e}")

        self._prev_failed_sigs = curr_sigs
        self._prev_detail = {to_signature(fc): fc for fc in current_failed}
        self._prev_reward = reward

    def harvest_skill(self):
        """After a successful run, abstract one reusable skill from the timeline."""
        if not self.store:
            return
        try:
            features = [f.code for f in self.canvas.features]
            if not features:
                return
            harvest_system = """
            You are a CAD knowledge curator. Given a CadQuery feature timeline that
            successfully produced a correct model, extract ONE reusable, parameterized
            skill that would help build similar models in the future. Generalize literal
            numbers into named parameters.
            Return a raw JSON object with keys: "name" (snake_case identifier),
            "goal_description" (one sentence), "signature" (parameter list like
            "(width, height)"), and "code_template" (the parameterized CadQuery snippet).
            Do not include markdown formatting.
            """
            resp = self._safe_call(
                self.client.models.generate_content,
                model=self.model_name,
                contents=f"Design prompt: {self.prompt}\nFeature timeline:\n" + "\n".join(features),
                config=types.GenerateContentConfig(
                    system_instruction=harvest_system,
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            data = json.loads(resp.text.strip())
            if data.get("name") and data.get("code_template"):
                self.store.record_skill(
                    data["name"], data.get("goal_description", ""),
                    data.get("signature", ""), data["code_template"],
                )
                self.log(f"[LOG] Harvested skill '{data['name']}' into learning store.")
        except Exception as e:
            self.log(f"[WARNING] Skill harvest failed: {e}")

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

    # ---------------------------------------------------- generation helpers
    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """Strip a leading/trailing markdown code fence from model output."""
        text = (text or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    def _select_color(self, prompt: str, default: str = "#FF9900") -> str:
        """Ask the model for a representative hex color for the design."""
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
                    response_mime_type="application/json",
                ),
            )
            return json.loads(resp.text.strip()).get("color", default)
        except Exception as e:
            self.log(f"[WARNING] Failed to auto-select color: {e}")
            return default

    def _candidate_generation_contents(self, prompt, constraints, image_path, strategy_hint, memory_preamble):
        """Build the content parts for one single-shot candidate generation."""
        text = f"Design Goal: {prompt}\nTarget constraints: {json.dumps(constraints)}"
        if strategy_hint:
            text += f"\n\n{strategy_hint}"
        if memory_preamble:
            text += f"\n\n# Reference (retrieved CadQuery API + prior lessons)\n{memory_preamble}"
        parts = [types.Part.from_text(text=text)]
        if image_path:
            img_path = Path(image_path)
            if img_path.exists():
                import mimetypes
                mime_type, _ = mimetypes.guess_type(img_path)
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type or "image/png"))
        return parts

    def _generate_candidate_script(self, prompt, constraints, image_path, strategy_hint, temperature, memory_preamble):
        contents = self._candidate_generation_contents(
            prompt, constraints, image_path, strategy_hint, memory_preamble
        )
        resp = self._safe_call(
            self.client.models.generate_content,
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=self.SINGLE_SHOT_SYSTEM,
                temperature=temperature,
            ),
        )
        return self._strip_code_fence(resp.text)

    def _execute_and_score_candidate(self, code: str, constraints: Dict[str, Any]) -> "candidate_search.Candidate":
        """Compile one candidate and wrap its result + graded score."""
        res = self.sandbox.execute(code)
        stl_file = self.sandbox.working_dir / "001.stl"
        generated_stl = str(stl_file) if (res.success and stl_file.exists()) else None
        reward, breakdown = RewardEngine.calculate_reward(
            res.success, res.metrics, constraints, generated_stl=generated_stl
        )
        err = ""
        if not res.success:
            err = (res.stderr or res.stdout or "compile failure").strip()[:200]
        return candidate_search.Candidate(
            strategy="",
            code=code,
            success=res.success,
            metrics=res.metrics,
            geom_score=breakdown.get("geom_score", 0.0),
            reward=reward,
            error=err,
        )

    def _run_candidate_search_impl(
        self, prompt, constraints, image_path, num_candidates, session_id
    ) -> Dict[str, Any]:
        """Generate N diverse candidates, rank them, and keep the best (EvoCAD-style)."""
        self.log(f"\n[LOG] Parallel candidate search: generating {num_candidates} diverse candidates...")
        if self.event_callback:
            self.event_callback({
                "type": "thought",
                "author": "MEDAOrchestrator",
                "content": f"Generating {num_candidates} candidate designs with different strategies and keeping the best.",
            })
        memory_preamble = self.build_memory_preamble(prompt)

        def generate_fn(name, hint, temp):
            if self.event_callback:
                self.event_callback({
                    "type": "thought",
                    "author": "CADModelerAgent",
                    "content": f"Candidate '{name}' (temp={temp}): {hint.split(':', 1)[-1].strip()[:80]}",
                })
            return self._generate_candidate_script(prompt, constraints, image_path, hint, temp, memory_preamble)

        def execute_fn(code):
            return self._execute_and_score_candidate(code, constraints)

        best, cands = candidate_search.run_candidate_search(
            num_candidates, generate_fn, execute_fn, self.log
        )

        success = bool(best and best.success)
        final_code = best.code if best else ""

        # Re-execute the winner so canonical artifacts (001.stl/step/png) and
        # iteration history reflect the selected candidate, not the last tried one.
        if best and best.code:
            self._execute_and_score_candidate(best.code, constraints)
            if success:
                self._save_success_artifacts(best.code, "candidate search")

        final_py_path = self.sandbox.working_dir / "001.py"
        with open(final_py_path, "w", encoding="utf-8") as f:
            f.write(final_code)

        color = self._select_color(prompt) if success else "#FF9900"

        if self.trajectory_memory:
            try:
                self.trajectory_memory.record_run(
                    prompt=prompt,
                    success=success,
                    metrics=best.metrics if best else None,
                    failed_constraints=[best.error] if (best and not success and best.error) else [],
                )
            except Exception as e:
                self.log(f"[WARNING] Failed to store trajectory memory: {e}")

        return {
            "success": success,
            "iterations": num_candidates,
            "final_reward": best.reward if best else 0.0,
            "final_code": final_code,
            "metrics": best.metrics if best else None,
            "failed_constraints": ([best.error] if (best and not success and best.error) else []),
            "color": color,
            "session_id": session_id,
            "candidates": [
                {"strategy": c.strategy, "success": c.success, "score": c.score,
                 "geom_score": c.geom_score, "error": c.error[:120]}
                for c in cands
            ],
        }

    def run_design_loop(
        self,
        prompt: str,
        constraints: Dict[str, Any],
        image_path: Optional[str] = None,
        max_iterations: int = 100,
        session_id: Optional[str] = None,
        keep_canvas: bool = False,
        fast_mode: bool = False,
        num_candidates: int = 1
    ) -> Dict[str, Any]:
        """Serialized public entry point (bug H4) that delegates to the impl."""
        with _run_lock:
            return self._run_design_loop_impl(prompt, constraints, image_path, max_iterations, session_id, keep_canvas, fast_mode, num_candidates)

    def _run_design_loop_impl(
        self,
        prompt: str,
        constraints: Dict[str, Any],
        image_path: Optional[str] = None,
        max_iterations: int = 100,
        session_id: Optional[str] = None,
        keep_canvas: bool = False,
        fast_mode: bool = False,
        num_candidates: int = 1
    ) -> Dict[str, Any]:
        """Runs the autonomous Chain-of-Thought tool execution loop until the design goal is achieved."""
        global _active_core
        _active_core = self
        self._reset_run_state()
        self.successful_compiles_count = 0
        
        self.log(f"\n[LOG] Initializing design loop for prompt: '{prompt}' (Fast Mode: {fast_mode})")
        self.log(f"[LOG] Target Constraints: {constraints}")
        if image_path:
            self.log(f"[LOG] Image Path: {image_path}")
        
        # Reset canvas state only if not keeping it
        if not keep_canvas:
            self.canvas = Canvas()
        else:
            # If keeping canvas but canvas is empty, and we have a compiled file, load it as Step 0
            if len(self.canvas.features) == 0:
                py_path = self.sandbox.working_dir / "001.py"
                if py_path.exists():
                    try:
                        with open(py_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        
                        # Strip imports and exports from loaded content
                        lines = content.splitlines()
                        clean_lines = []
                        in_model_gen = False
                        
                        for line in lines:
                            stripped = line.strip()
                            if stripped == "# === MODEL GENERATION ===":
                                in_model_gen = True
                                continue
                            elif stripped == "# === EXPORTS ===":
                                in_model_gen = False
                                break
                            if in_model_gen:
                                clean_lines.append(line)
                                
                        clean_code = "\n".join(clean_lines).strip()
                        if clean_code:
                            # Extract parameters if present
                            in_params = False
                            for line in lines:
                                stripped = line.strip()
                                if stripped == "# === PARAMETERS ===":
                                    in_params = True
                                    continue
                                elif stripped == "# === MODEL GENERATION ===":
                                    in_params = False
                                    break
                                if in_params:
                                    import re
                                    match = re.search(r"([a-zA-Z_]+)\s*=\s*([-+]?[0-9.]+)", line)
                                    if match:
                                        self.canvas.add_parameter(match.group(1), float(match.group(2)))
                            
                            self.canvas.add_feature(clean_code, "Imported base model")
                    except Exception as e:
                        self.log(f"[WARNING] Failed to load existing 001.py into canvas: {e}")
        self.prompt = prompt
        
        # Extract/infer constraints from prompt using a simple model call
        inferred_constraints = {}
        try:
            # NOTE (bug H3): we deliberately do NOT infer exact face/edge/vertex
            # counts here. Those counts shift unpredictably with fillets, holes,
            # and patterns, so a guessed integer target made most designs
            # unsatisfiable under the strict equality gate. We only infer
            # tolerance-checked metrics (volume, center of mass); shape
            # correctness is enforced by the visual critic instead.
            system_extract = """
            You are a CAD parameter extraction assistant. Analyze the user design prompt and extract ONLY clearly-stated or directly-calculable target constraints for these tolerance-checked B-Rep metrics:
            - "volume": target total volume in mm^3 (float)
            - "center_of_mass": [x, y, z] expected center of mass coordinates (list of floats)

            Do NOT guess face, edge, or vertex counts. Return the output as a valid JSON object with no markdown.
            If nothing can be confidently calculated, return {}.
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

        # Parallel candidate search (EvoCAD-style): generate N diverse single-shot
        # candidates and keep the best by compile + geometry + simplicity score.
        if num_candidates and num_candidates > 1:
            try:
                result = self._run_candidate_search_impl(
                    prompt, self.constraints, image_path, int(num_candidates), session_id
                )
            finally:
                _active_core = None
            return result

        if fast_mode:
            self.log("\n[LOG] Fast Mode: Generating CAD model in a single-shot execution...")
            if self.event_callback:
                self.event_callback({
                    "type": "thought",
                    "author": "MEDAOrchestrator",
                    "content": "Running in Fast Mode (Single-Shot generation). Bypassing multi-agent loop."
                })
            
            parts = [types.Part.from_text(text=f"Design Goal: {prompt}\nTarget constraints: {json.dumps(self.constraints)}")]
            if image_path:
                img_path = Path(image_path)
                if img_path.exists():
                    import mimetypes
                    mime_type, _ = mimetypes.guess_type(img_path)
                    with open(img_path, "rb") as f:
                        img_bytes = f.read()
                    parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type or "image/png"))
            
            system_instruction = self.SINGLE_SHOT_SYSTEM

            attempts = 3
            current_code = ""
            error_log = ""
            success = False
            last_res = None
            iteration = 0
            
            for attempt in range(1, attempts + 1):
                iteration += 1
                self.log(f"[LOG] Direct generation attempt {attempt}/{attempts}...")
                if self.event_callback:
                    self.event_callback({
                        "type": "thought",
                        "author": "CADModelerAgent",
                        "content": f"Writing and compiling CAD script (Attempt {attempt})..."
                    })
                
                if attempt == 1:
                    contents = parts
                else:
                    contents = [
                        f"Your previous code failed compilation. Here is the code you generated:\n\n```python\n{current_code}\n```\n\nAnd here is the execution error log:\n\n{error_log}\n\nPlease fix the errors and rewrite the entire script. Ensure it is complete and self-contained."
                    ]
                    
                resp = self._safe_call(
                    self.client.models.generate_content,
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.2 if attempt > 1 else 0.0
                    )
                )
                
                text = resp.text.strip()
                if text.startswith("```"):
                    lines = text.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    text = "\n".join(lines).strip()
                
                current_code = text
                
                # Execute in sandbox
                res = self.sandbox.execute(current_code)
                last_res = res
                
                if res.success:
                    self.log("[LOG] Fast Mode generation success! Script compiled successfully.")
                    success = True
                    self._save_success_artifacts(current_code, "Fast Mode")
                    break
                else:
                    error_log = res.stderr if res.stderr else res.stdout
                    self.log(f"[WARNING] Compilation failed: {error_log.strip()}")
            
            # Select color
            suggested_color = self._select_color(prompt) if success else "#FF9900"

            # Save files
            final_py_path = self.sandbox.working_dir / "001.py"
            with open(final_py_path, "w", encoding="utf-8") as f:
                f.write(current_code)
            
            # Clean global pointer
            _active_core = None
            
            return {
                "success": success,
                "iterations": iteration,
                "final_reward": 1.0 if success else 0.0,
                "final_code": current_code,
                "metrics": last_res.metrics if last_res else None,
                "failed_constraints": [error_log] if not success else [],
                "color": suggested_color,
                "session_id": session_id
            }

        # Instantiate the retrying Gemini wrapper for ADK
        model_wrapper = RetryingGemini(model=self.model_name)
        if getattr(self, "_api_key_override", None):
            model_wrapper._api_key = self._api_key_override

        # Inject retrieved skills + lessons from past runs so the modeler starts
        # informed and avoids repeating known mistakes (self-learning loop).
        memory_preamble = self.build_memory_preamble(prompt)
        modeler_instruction = self.MODELER_INSTRUCTION
        critic_instruction = self.CRITIC_INSTRUCTION
        
        if memory_preamble:
            modeler_instruction = (
                modeler_instruction
                + "\n\n# PRIOR EXPERIENCE (retrieved from the learning store)\n"
                + memory_preamble
                + "\n\nReuse the skills above where helpful and heed the lessons.\n"
            )
            critic_instruction = (
                critic_instruction
                + "\n\nNote: an image of the current rendered model is attached each turn; "
                "use it to visually verify part placement and report misalignments."
            )
            self.log("[LOG] Injected prior lessons/skills into modeler context.")

        # Best-effort callback that attaches the latest render to each agent turn
        # so the LLM can visually course-correct.
        def _attach_render(callback_context, llm_request):
            try:
                if self.last_png_bytes and getattr(llm_request, "contents", None):
                    part = types.Part.from_bytes(data=self.last_png_bytes, mime_type="image/png")
                    llm_request.contents[-1].parts.append(part)
            except Exception:
                pass
            return None

        # Setup specialized sub-agents
        modeler_agent = LlmAgent(
            name="CADModelerAgent",
            model=model_wrapper,
            description="CAD modeler specialist that defines parameters and updates timeline features",
            instruction=modeler_instruction,
            tools=[add_parameter, set_parameter, add_feature, modify_feature,
                   remove_feature, lookup_cadquery_api, inspect_current_model],
            before_model_callback=_attach_render,
        )

        critic_agent = LlmAgent(
            name="VisualCriticAgent",
            model=model_wrapper,
            description="Verification specialist that compiles CAD models and runs visual checks",
            instruction=critic_instruction,
            tools=[run_cad_execution],
            before_model_callback=_attach_render,
        )

        # Setup main orchestrator agent
        orchestrator_agent = LlmAgent(
            name="MEDAOrchestrator",
            model=model_wrapper,
            description="Lead orchestrator agent that coordinates the design loop using handoffs",
            instruction=self.ORCHESTRATOR_INSTRUCTION,
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

        # Build parts for initial message. (The retrieved memory/skills/API docs
        # are already injected into the modeler/critic instructions above, so the
        # opening user message stays concise — the previously-referenced
        # `memory_guidance` variable never existed and crashed this path.)
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
        if not session_id:
            session_id = f"session_{int(time.time())}"

        current_reward = 0.0
        last_execution_result = {}
        iteration = 0
        exec_turns = 0
        event_cap = max(50, max_iterations * 30)

        # Define async execution function to run the generator
        async def run_agent():
            nonlocal current_reward, last_execution_result, iteration, exec_turns
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
                            if self.event_callback:
                                self.event_callback({
                                    "type": "thought",
                                    "author": event.author,
                                    "content": part.text
                                })
                
                # Log agent transfers
                if event.actions and event.actions.transfer_to_agent:
                    transfer_msg = f"Handoff from '{event.author}' to '{event.actions.transfer_to_agent}'"
                    self.log(f"\n[TRANSFER] {transfer_msg}")
                    if self.event_callback:
                        self.event_callback({
                            "type": "handoff",
                            "author": event.author,
                            "content": transfer_msg
                        })
                
                # Log function call triggers
                func_calls = event.get_function_calls()
                if func_calls:
                    for call in func_calls:
                        call_msg = f"Executing: {call.name} with arguments: {call.args}"
                        self.log(f"[{event.author} TOOL_CALL] {call_msg}")
                        
                        code = None
                        title = call.name
                        if call.name in ["add_feature", "modify_feature"]:
                            code = call.args.get("code")
                            title = call.args.get("description", "Adding feature step")
                            
                        if self.event_callback:
                            self.event_callback({
                                "type": "tool_call",
                                "author": event.author,
                                "name": call.name,
                                "title": title,
                                "content": call_msg,
                                "code": code
                            })
                
                # Log function responses and inspect run_cad_execution outputs
                func_responses = event.get_function_responses()
                if func_responses:
                    for resp in func_responses:
                        val = resp.response.get("result") if resp.response else ""
                        res_str = str(val) if val is not None else ""
                        self.log(f"[{event.author} TOOL_RESPONSE] {res_str[:300]}...")
                        
                        reward = None
                        failed_constraints = []
                        if resp.name == "run_cad_execution":
                            exec_turns += 1
                            try:
                                exec_data = json.loads(res_str)
                                current_reward = exec_data.get("reward", 0.0)
                                last_execution_result = exec_data
                                reward = current_reward
                                failed_constraints = exec_data.get("failed_constraints", [])
                            except Exception:
                                pass
                                
                        if self.event_callback:
                            self.event_callback({
                                "type": "tool_response",
                                "author": event.author,
                                "name": resp.name,
                                "content": f"{res_str[:300]}...",
                                "reward": reward,
                                "failed_constraints": failed_constraints
                            })

                if current_reward == 1.0:
                    self.log(f"[LOG] Success reached (reward=1.0) after {exec_turns} execution turn(s).")
                    break
                if exec_turns >= max_iterations:
                    self.log(f"[WARNING] Reached max_iterations ({max_iterations}); stopping loop.")
                    break
                if iteration >= event_cap:
                    self.log(f"[WARNING] Event cap ({event_cap}) reached; stopping loop.")
                    break

        # Run the async loop synchronously
        import asyncio
        asyncio.run(run_agent())

        # Auto-select suitable color hex based on prompt
        suggested_color = "#FF9900"
        if current_reward == 1.0:
            self.harvest_skill()
            suggested_color = self._select_color(prompt)
            self.log(f"[LOG] Auto-selected color {suggested_color} for prompt '{prompt}'")

        final_code = self.canvas.to_python_code()
        
        # Save final files to NewCADs
        final_py_path = self.sandbox.working_dir / "001.py"
        with open(final_py_path, "w", encoding="utf-8") as f:
            f.write(final_code)
            
        memory_counts = self.store.counts() if self.store else {}

        # Compile and save diagnostic snapshot JSON
        diagnostic = {
            "prompt": prompt,
            "constraints": self.constraints,
            "design_iterations": exec_turns,
            "events": iteration,
            "final_reward": current_reward,
            "final_code": final_code,
            "metrics": last_execution_result.get("metrics"),
            "failed_constraints": last_execution_result.get("failed_constraints", []),
            "suggested_color": suggested_color,
            "memory_counts": memory_counts,
        }
        diagnostic_path = self.sandbox.working_dir / "diagnostic.json"
        with open(diagnostic_path, "w", encoding="utf-8") as f:
            json.dump(diagnostic, f, indent=2)

        if self.trajectory_memory:
            try:
                self.trajectory_memory.record_run(
                    prompt=prompt,
                    success=current_reward == 1.0,
                    metrics=last_execution_result.get("metrics"),
                    failed_constraints=last_execution_result.get("failed_constraints", []),
                )
                self.log("[LOG] Stored trajectory memory for future self-improvement.")
            except Exception as e:
                self.log(f"[WARNING] Failed to store trajectory memory: {e}")

        # Clean global pointer
        _active_core = None
            
        return {
            "success": current_reward == 1.0,
            "iterations": exec_turns,
            "events": iteration,
            "final_reward": current_reward,
            "final_code": final_code,
            "metrics": last_execution_result.get("metrics"),
            "failed_constraints": last_execution_result.get("failed_constraints", []),
            "color": suggested_color,
            "session_id": session_id,
            "memory_counts": memory_counts,
        }
