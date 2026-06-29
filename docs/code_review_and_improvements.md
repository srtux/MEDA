# Code Review & Architectural Improvements for MEDA

Based on the analysis of `task-304.log` and the current codebase, here are the key findings, issues identified, and recommended structural improvements to make the multi-agent system more robust, faster, and cost-effective.

---

## 1. Key Log Analysis Findings

### ✅ Auto-Debugging and Execution Works
The execution pipeline succeeded:
*   The **CAD Script Writer** initially generated code containing `show_object(boxy_cat)`, which failed because `show_object` is only defined in standard CQ-Editor and not in standalone Python execution.
*   The **Executor** successfully captured the `NameError` and sent the stack trace back to the group chat.
*   The **CAD Script Writer** correctly processed the error, removed `show_object`, replaced it with `show()` from `ocp_vscode`, and successfully re-executed (exit code `0`), generating `001.stl` and `001.step`.

### ❌ Headless Screenshot Failure (Critical Loophole)
The generated script called `save_screenshot("001.png")` from `ocp_vscode`. This failed with:
`Warning: Screenshot not found in 2 seconds, aborting`
*   **Reason**: `ocp_vscode` depends on an active, local VS Code editor session to render the viewer and capture a viewport screenshot. Running headlessly in the background environment prevents this.
*   **Consequence**: The **CAD Image Reviewer** (the multimodal visual agent) depends on this PNG to run. Because the PNG is missing, visual validation fails or is bypassed entirely.

### 🐛 CAD Image Reviewer Omission Bug
In `streamlitapp.py` (lines 174–184), the `cad_image_reviewer` agent (index 5) is completely omitted from the agent lists passed to the chats in both multimodal and text-only modes.
*   This was likely a quick bypass introduced to avoid crashes since the screenshot file (`001.png`) was never successfully generated.

### 🐢 Inefficient Speaker Selection (High Latency & Costs)
The group chat uses `speaker_selection_method="auto"`. 
*   **Reason**: Under the hood, AutoGen makes an LLM call at *every turn* just to decide which agent should speak next (e.g., calling Gemini to decide that `Executor` should run after `CAD_Script_Writer`).
*   **Consequence**: Adds **2–5 seconds of latency per message** and consumes substantial API tokens, making the system slower and much more expensive to run.

---

## 2. Proposed Improvements

### 🛠️ Improvement 1: Dynamic Headless Screenshot Generation (Implemented & Verified)
We installed `open3d` in the environment and successfully verified a script that loads `001.stl` and saves a screenshot `001.png` completely headlessly.

We should modify `utils/get_image_info.py` to automatically trigger this screenshot generator from the latest STL file before trying to read the image:

```python
def get_latest_stl(directory="NewCADs"):
    from pathlib import Path
    try:
        dir_path = Path(directory).resolve()
        stl_files = list(dir_path.glob('*.stl'))
        if not stl_files:
            return None
        return max(stl_files, key=lambda x: x.stat().st_ctime)
    except Exception:
        return None

def get_image_info(prompt, cad_working_dir="NewCADs"):
    # Generate PNG on the fly from the latest STL file
    latest_stl = get_latest_stl(cad_working_dir)
    if latest_stl:
        png_path = latest_stl.with_suffix('.png')
        try:
            import open3d as o3d
            mesh = o3d.io.read_triangle_mesh(str(latest_stl))
            if not mesh.is_empty():
                mesh.compute_vertex_normals()
                vis = o3d.visualization.Visualizer()
                vis.create_window(window_name="CAD Viewer", width=800, height=600, visible=False)
                vis.add_geometry(mesh)
                vis.update_geometry(mesh)
                vis.poll_events()
                vis.capture_screen_image(str(png_path), do_render=True)
                vis.destroy_window()
        except Exception as e:
            print(f"[WARNING] Headless screenshot generation failed: {e}")
            
    # Proceed to load image and send to Gemini/OpenAI...
```

### 🛠️ Improvement 2: Restore Visual Validation Agent
Add the `cad_image_reviewer` agent back to `multimodal_agents` and `text_agents` in `streamlitapp.py`:

```python
    multimodal_agents = [
        agents_list[0], # User
        agents_list[1], # Design_Expert
        agents_list[2], # CAD_Script_Writer
        agents_list[3], # Executor
        agents_list[4], # Script_Execution_Reviewer
        agents_list[5], # CAD_Image_Reviewer (Restored!)
        agents_list[6], # CAD_Data_Reviewer
    ]
```

### 🛠️ Improvement 3: Deterministic Speaker Selection (State Transition Graph)
Instead of LLM-based `"auto"` speaker selection, we can configure a custom transition graph to define the sequence of roles. 

Since the flow of control is strictly defined:
1.  `User` → `Design_Expert`
2.  `Design_Expert` → `CAD_Script_Writer`
3.  `CAD_Script_Writer` → `Executor`
4.  `Executor` → `Script_Execution_Reviewer`
5.  `Script_Execution_Reviewer` → `CAD_Image_Reviewer` (if success) OR `CAD_Script_Writer`/`Design_Expert` (if fail)
6.  `CAD_Image_Reviewer` → `Design_Expert` (if fail) OR `User`/`TERMINATE` (if success)

We can define a deterministic transition function in AutoGen to enforce this. This will **eliminate the speaker selection LLM overhead entirely**, saving up to 30% on token costs and execution time.

---

## 3. Resolution: Unified Reasoning Core Redesign (Checkpoint 3)

We resolved the issues identified above by replacing the AutoGen multi-agent group chat system with a unified Reasoning Core (`gemini-3.5-flash`) coupled with a programmatic sandbox and gating reward engine:

1. **Deterministic Speaker Selection Overhead**: Completely eliminated by collapsing the multi-agent chat graph into a single autonomous CoT loop. The model directly calls tools (`add_parameter`, `add_feature`, `run_cad_execution`), saving 100% of speaker selection LLM latency and costs.
2. **Headless Screenshot & Image Reviewer Restored**: Implemented a headless `open3d` + `PIL` rendering script ([utils/capture_screenshot.py](file:///Users/summitt/work/MEDA/utils/capture_screenshot.py)) that generates a 4-view orthographic drawing collage. Integrated a visual validator critique step directly inside the reward engine using `gemini-3.5-flash` to inspect visual correctness.
3. **Parametric Timeline History**: Implemented a centralized `Canvas` shape parameters timeline to prevent regenerating code from scratch upon compilation errors, allowing surgical delta updates.
4. **Transient API Error Resiliency**: Wrapped all API calls in an exponential backoff retry loop targeting 503 and 429 errors.

