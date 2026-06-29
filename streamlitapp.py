"Run this script to start the Streamlit app to create CAD models"
import os
from dotenv import load_dotenv
load_dotenv()
import time
from pathlib import Path

import streamlit as st
from streamlit_stl import stl_from_file

from core.reasoning_core import ReasoningCADCore
from streamlit_utils.file_handler import FileHandler
from streamlit_utils.parameter_handler import ParameterHandler
from streamlit_utils.state_manager import StateManager
from config.llm_config import LLMConfigSelector, process_custom_llm_config


def render_custom_llm_config():
    """Render custom LLM configuration controls in a consistent field order."""
    model_name = st.text_input("Model name", placeholder="e.g. gemini-2.5-pro")
    api_type = st.text_input("API type", placeholder="google, azure, openai-compatible")
    api_key = st.text_input("API key", type="password")
    base_url = st.text_input("Base URL (optional)") or None
    api_version = st.text_input("API version (optional)") or None
    return model_name, api_type, api_key, base_url, api_version


def render_llm_config_sidebar():
    """Render the LLM configuration controls in the sidebar."""
    with st.sidebar:
        st.title("⚙️ Model Studio")
        st.caption("Choose the reasoning model that will plan, code, execute, and critique CAD geometry.")

        selector = LLMConfigSelector()
        models_to_select = ["Default Gemini",
                            "Default GPT-4o",
                            "Default O1", "Text LLM", "Multimodal LLM"]
        option_selected = st.selectbox("Select model type", models_to_select)
        st.session_state.selected_model = option_selected

        if st.session_state.selected_model == "Default Gemini":
            model_info = selector.get_default_model_info("gemini-3.5-flash")
            api_key = os.environ.get(model_info["api_key"])
            if not api_key:
                st.warning(f"Please configure {model_info['api_key']} in your environment, or select another model.")
                st.session_state.config_created = False
            else:
                config = {
                    "model": model_info["model"],
                    "api_key": api_key,
                    "api_type": model_info["api_type"]
                }
                st.session_state.llm_config = config
                st.session_state.config_created = True
                st.success("Configuration created successfully!")

        if st.session_state.selected_model == "Default GPT-4o":
            model_info = selector.get_default_model_info("gpt-4o")
            api_key = os.environ.get(model_info["api_key"])
            base_url = os.environ.get(model_info["base_url"])
            if not api_key or not base_url:
                st.warning(f"Please configure {model_info['api_key']} and {model_info['base_url']} in your environment, or select another model.")
                st.session_state.config_created = False
            else:
                config = {
                    "model": model_info["model"],
                    "api_key": api_key,
                    "api_type": model_info["api_type"],
                    "base_url": base_url,
                    "api_version": model_info["api_version"]
                }
                st.session_state.llm_config = config
                st.session_state.config_created = True
                st.success("Configuration created successfully!")

        if st.session_state.selected_model == "Default O1":
            model_info = selector.get_default_model_info("o1")
            api_key = os.environ.get(model_info["api_key"])
            base_url = os.environ.get(model_info["base_url"])
            if not api_key or not base_url:
                st.warning(f"Please configure {model_info['api_key']} and {model_info['base_url']} in your environment, or select another model.")
                st.session_state.config_created = False
            else:
                config = {
                    "model": model_info["model"],
                    "api_key": api_key,
                    "api_type": model_info["api_type"],
                    "base_url": base_url,
                    "api_version": model_info["api_version"]
                }
                st.session_state.llm_config = config
                st.session_state.config_created = True
                st.success("Configuration created successfully!")

        if st.session_state.selected_model == "Text LLM":
            available_models = selector.get_available_models(False)
            available_models.append("Custom Model")
            model_name = st.selectbox("Text only LLMs", available_models)
            if model_name == "Custom Model":
                model_name, api_type, api_key, base_url, api_version = render_custom_llm_config()
                config_custom = process_custom_llm_config(
                    model_name, api_type, api_key, base_url, api_version)
                st.session_state.llm_config = config_custom
                st.session_state.config_created = True
                st.success("Configuration created successfully!")
            else:
                model_info = selector.get_model_info(model_name)
                api_key_from_env = selector.get_api_key_from_env(model_name)
                use_env_key = st.checkbox(
                    "Use API key from environment", value=bool(api_key_from_env))
                if use_env_key and api_key_from_env:
                    api_key = api_key_from_env
                    st.success("Using API key from environment")
                else:
                    api_key = st.text_input("API Key", type="password")
                config = selector.create_config(model_name, api_key)
                st.session_state.llm_config = config
                st.session_state.config_created = True
                st.success("Configuration created successfully!")

        if st.session_state.selected_model == "Multimodal LLM":
            available_models = selector.get_available_models(True)
            available_models.append("Custom Model")
            model_name = st.selectbox("Multimodal LLMs", available_models)
            if model_name == "Custom Model":
                model_name, api_type, api_key, base_url, api_version = render_custom_llm_config()
                config_custom = process_custom_llm_config(
                    model_name, api_type, api_key, base_url, api_version)
                st.session_state.llm_config = config_custom
                st.session_state.config_created = True
                st.success("Configuration created successfully!")

            else:
                model_info = selector.get_model_info(model_name)
                api_key_from_env = selector.get_api_key_from_env(model_name)
                use_env_key = st.checkbox(
                    "Use API key from environment", value=bool(api_key_from_env))
                if use_env_key and api_key_from_env:
                    api_key = api_key_from_env
                    st.success("Using API key from environment")
                else:
                    api_key = st.text_input("API Key", type="password")
                config = selector.create_config(model_name, api_key)
                st.session_state.llm_config = config
                st.session_state.config_created = True
                st.success("Configuration created successfully!")


def initialize_session_state():
    "Intialize default state of streamlit UI"
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "Default Gemini 3.5 Flash"
    if 'llm_config' not in st.session_state:
        api_key = os.environ.get("GEMINI_API_KEY")
        st.session_state.llm_config = {
            "model": "gemini-3.5-flash",
            "api_key": api_key,
            "api_type": "google"
        }
    if 'config_created' not in st.session_state:
        st.session_state.config_created = bool(st.session_state.llm_config.get("api_key"))
    default_state = StateManager.get_default_state()
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.stl_timestamp is None:
        st.session_state.stl_timestamp = time.time()
    if 'log_history' not in st.session_state:
        st.session_state.log_history = ""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'core' not in st.session_state:
        st.session_state.core = None
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'fast_mode' not in st.session_state:
        st.session_state.fast_mode = False
    if 'num_candidates' not in st.session_state:
        st.session_state.num_candidates = 1
    if 'selected_iteration_label' not in st.session_state:
        st.session_state.selected_iteration_label = "Latest"


def render_parameter_controls(python_file_path):
    "Render the parameter controls for the CAD model"
    st.markdown('<div class="section-title">Parametric controls</div><div class="section-subtitle">Tune generated dimensions and regenerate the CAD asset.</div>', unsafe_allow_html=True)

    if not python_file_path or not Path(python_file_path).exists():
        return

    handler = ParameterHandler()
    current_params = handler.extract_parameters(python_file_path)

    if not current_params:
        return

    new_params = {}
    for param_name, param_value in current_params.items():
        new_value = st.number_input(
            f"{param_name}",
            value=float(param_value),
            step=0.1,
            format="%.3f",
            key=f"param_{param_name}"
        )
        new_params[param_name] = new_value

    if st.button("Regenerate CAD", key="regenerate_cad"):
        if handler.update_python_file(python_file_path, new_params):
            if handler.execute_python_file(python_file_path):
                st.session_state.stl_timestamp = time.time()
                st.success("CAD model regenerated successfully!")
                st.rerun()


def trigger_generation(prompt: str):
    "Trigger the background design loop by setting session state variables and rerunning"
    if not st.session_state.messages or st.session_state.messages[-1]["content"] != prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.prompt = prompt
    st.session_state.is_generating = True
    st.session_state.active_prompt = prompt
    st.session_state.selected_iteration_label = "Latest"
    st.rerun()


def render_events_status(events: list):
    "Render ADK structured reasoning events inside a status timeline"
    if not events:
        return
        
    with st.status("🛠️ MEDA Design Execution Timeline", expanded=True):
        for ev in events:
            role = ev.get("author", "Agent")
            ev_type = ev.get("type")
            
            if ev_type == "handoff":
                st.markdown(f"🔄 **Handoff:** {ev['content']}")
            elif ev_type == "tool_call":
                if ev.get("code"):
                    with st.expander(f"💻 *{role} code addition: {ev['title']}*", expanded=True):
                        st.code(ev["code"], language="python")
                else:
                    st.markdown(f"🔧 **Tool Call:** `{ev['name']}`")
            elif ev_type == "tool_response":
                if ev.get("reward") is not None:
                    reward = ev["reward"]
                    if reward == 1.0:
                        st.markdown("🎯 **Visual Critic Verification:** :green[Success (Reward: 1.0)]")
                    else:
                        st.markdown("⚠️ **Visual Critic Verification:** :red[Constraint Violation (Reward: 0.0)]")
                        if ev.get("failed_constraints"):
                            for err in ev["failed_constraints"]:
                                st.write(f"- ❌ *{err}*")
                else:
                    # Truncate raw response text to keep UI clean
                    trunc = ev["content"][:150] + ("..." if len(ev["content"]) > 150 else "")
                    st.write(f"📝 **Response:** {trunc}")
            elif ev_type == "thought":
                content = ev["content"].strip()
                if content:
                    st.write(f"💡 *{role}:* {content}")


def render_chat_workspace():
    "Render the main chat interface and prompt input area"
    st.markdown('<div class="section-title">💬 CAD Copilot</div><div class="section-subtitle">Describe a model, attach a reference sketch, or refine the current geometry.</div>', unsafe_allow_html=True)
    
    # Scrollable chat messages container
    chat_container = st.container(height=800, border=True)
    with chat_container:
        if not st.session_state.messages:
            with st.chat_message("assistant"):
                st.write("Hello! I am MEDA, your autonomous CAD design agent. What would you like to design today?")
                st.write("**Here are some suggestions to get started:**")
                examples = [
                    {"label": "📦 Simple Box with Hole", "prompt": "A box with a through hole in the center."},
                    {"label": "🧪 Cylindrical Pipe (50mm/40mm)", "prompt": "Create a pipe of outer diameter 50mm and inside diameter 40mm."},
                    {"label": "💿 Parametric Multi-hole Plate", "prompt": "Create a circular plate of radius 2mm and thickness 0.125mm with four holes of radius 0.25mm patterned at distance of 1.5mm from the centre along the axes."}
                ]
                for ex in examples:
                    if st.button(ex["label"], key=f"chat_example_{ex['label']}", use_container_width=True):
                        st.session_state._clicked_prompt = ex["prompt"]
                        st.rerun()
        else:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
                    if "adk_events" in msg and msg["adk_events"]:
                        render_events_status(msg["adk_events"])
                    if "logs" in msg and msg["logs"]:
                        with st.expander("🛠️ View Agent Debug Trace Logs", expanded=False):
                            st.code(msg["logs"], language="log")
            
            # Show live generating state in bubble
            if st.session_state.get("is_generating", False):
                with st.chat_message("assistant"):
                    st.write("🤖 Thinking & generating CAD model...")
                    
                    # Placeholder for ADK structured visual timeline
                    st.session_state.active_status_placeholder = st.empty()
                    
                    # Placeholder for raw logs in a collapsed state
                    active_expander = st.expander("Debug Logs (Live)", expanded=False)
                    st.session_state.active_log_placeholder = active_expander.empty()

    # Render Fast Mode toggle, candidate-search control, and attachment caption.
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 1])
    with ctrl_col1:
        st.session_state.fast_mode = st.toggle(
            "⚡ Fast Mode (Single-Shot)",
            value=st.session_state.get("fast_mode", False),
            help="Generate the script in one shot, bypassing the slow multi-agent critique loops. Best for simple shapes."
        )
    with ctrl_col2:
        st.session_state.num_candidates = st.slider(
            "🧪 Parallel candidates",
            min_value=1, max_value=5,
            value=st.session_state.get("num_candidates", 1),
            help="Generate this many candidate designs with different modeling strategies and automatically keep the best (compile + geometry + simplicity). 1 = off. Overrides the loop mode when > 1."
        )
    with ctrl_col3:
        uploaded_file = st.session_state.get("drawing_uploader")
        if uploaded_file is not None:
            st.caption(f"📎 Attached drawing: `{uploaded_file.name}`")

    # Unified Google Gemini-style Prompt Box Capsule
    with st.container(border=True):
        if "chat_input_val" not in st.session_state:
            st.session_state.chat_input_val = ""
            
        def submit_prompt():
            val = st.session_state.prompt_input_field
            if val.strip():
                st.session_state.chat_input_val = val
                st.session_state.prompt_input_field = ""

        # Single row columns layout inside the card: Upload (+), Reset (🔄), Prompt (input), Send (✈️)
        in_col1, in_col2, in_col3, in_col4 = st.columns([0.6, 0.6, 7.8, 1.0])
        
        with in_col1:
            uploaded_file = st.file_uploader(
                "Attach", 
                type=["png", "jpg", "jpeg"],
                label_visibility="collapsed",
                key="drawing_uploader"
            )
            if uploaded_file is not None:
                if not st.session_state.current_image_path or not Path(st.session_state.current_image_path).exists():
                    st.session_state.current_image_path = FileHandler.save_uploaded_file(uploaded_file)
            else:
                if st.session_state.current_image_path:
                    st.session_state.current_image_path = None
                    
        with in_col2:
            if st.button("🔄", key="reset_session_btn", help="Reset Session", use_container_width=True):
                st.session_state.messages = []
                st.session_state.core = None
                st.session_state.session_id = None
                st.session_state.generated_py_file = None
                st.session_state.current_stl_path = 'docs/MEDA.stl'
                st.session_state.log_history = ""
                st.session_state.prompt = ""
                st.session_state.current_image_path = None
                st.session_state.selected_iteration_label = "Latest"
                st.rerun()
                
        with in_col3:
            st.text_input(
                "Prompt",
                placeholder="Ask me to design or modify a model...",
                label_visibility="collapsed",
                key="prompt_input_field",
                on_change=submit_prompt
            )
            
        with in_col4:
            send_clicked = st.button("➤", key="send_btn", help="Send Prompt", use_container_width=True)
            if send_clicked:
                val = st.session_state.prompt_input_field
                if val.strip():
                    st.session_state.chat_input_val = val
                    st.session_state.prompt_input_field = ""
                    st.rerun()

    # Trigger generation from suggestion click
    if st.session_state.get("_clicked_prompt"):
        p = st.session_state._clicked_prompt
        del st.session_state._clicked_prompt
        trigger_generation(p)

    # Trigger generation from text input
    if st.session_state.get("chat_input_val"):
        p = st.session_state.chat_input_val
        st.session_state.chat_input_val = ""
        trigger_generation(p)


def get_iteration_stl_files(working_dir):
    import re
    if not working_dir or not Path(working_dir).exists():
        return []
    p = Path(working_dir)
    files = list(p.glob("001_iter_*.stl"))
    def get_iter_num(f):
        match = re.search(r"001_iter_(\d+)\.stl", f.name)
        return int(match.group(1)) if match else 0
    files.sort(key=get_iter_num)
    return [str(f) for f in files]


def render_right_column():
    "Render the 3D model visualizer, download controls, parameters, and 3D viewer style selectors"
    st.markdown('<div class="section-title">📐 Live 3D Preview</div><div class="section-subtitle">Inspect iterations, adjust viewer materials, and export production-ready files.</div>', unsafe_allow_html=True)
    
    # Check if a model is available to preview (either default or generated)
    stl_file = st.session_state.current_stl_path
    if stl_file and Path(stl_file).exists():
        working_dir = Path(stl_file).parent
        iter_files = get_iteration_stl_files(working_dir)
        
        if iter_files:
            options = ["Latest"] + [f"Iteration {i+1}" for i in range(len(iter_files))]
            selected_option = st.select_slider(
                "📜 Design Iteration History",
                options=options,
                value=st.session_state.get("selected_iteration_label", "Latest"),
                help="Slide through intermediate design iterations to visualize progress and improvements."
            )
            st.session_state.selected_iteration_label = selected_option
            
            if selected_option == "Latest":
                st.session_state.current_stl_path = str(working_dir / "001.stl")
                if st.session_state.generated_py_file:
                    st.session_state.generated_py_file = str(working_dir / "001.py")
            else:
                iter_idx = options.index(selected_option) - 1
                st.session_state.current_stl_path = iter_files[iter_idx]
                if st.session_state.generated_py_file:
                    st.session_state.generated_py_file = iter_files[iter_idx].replace(".stl", ".py")
            
            st.write("---")

        # 3D Viewer Style Controls right above the viewer!
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 2, 2])
        with ctrl_col1:
            color = st.color_picker("Color", value=st.session_state.color, label_visibility="collapsed")
        with ctrl_col2:
            material = st.selectbox(
                "Mode",
                ["material", "flat", "wireframe"],
                index=["material", "flat", "wireframe"].index(st.session_state.material),
                label_visibility="collapsed"
            )
        with ctrl_col3:
            auto_rotate = st.toggle("Auto-Rotate", value=st.session_state.auto_rotate)
        
        # Sliders below selectors
        opacity_col, height_col = st.columns(2)
        with opacity_col:
            opacity = st.slider("Opacity", min_value=0.0, max_value=1.0, value=st.session_state.opacity)
        with height_col:
            height = st.slider("Canvas Height", min_value=100, max_value=800, value=st.session_state.height)
            
        st.session_state.color = color
        st.session_state.material = material
        st.session_state.auto_rotate = auto_rotate
        st.session_state.opacity = opacity
        st.session_state.height = height

        st.write("---")
        render_stl_viewer()
        
        # Only render download buttons and parameter controls if a file has actually been generated
        if st.session_state.generated_py_file:
            render_download_buttons()
            render_parameter_controls(st.session_state.generated_py_file)
    else:
        st.info("No model is available for preview. Use the CAD Chat workspace on the left to design a model.")


def render_stl_viewer():
    "Render the STL viewer"
    viewer_key = f'stl_viewer_{st.session_state.current_stl_path}_{st.session_state.stl_timestamp}'
    stl_from_file(
        file_path=st.session_state.current_stl_path,
        color=st.session_state.color,
        material=st.session_state.material,
        auto_rotate=st.session_state.auto_rotate,
        opacity=st.session_state.opacity,
        height=st.session_state.height,
        shininess=100,
        cam_v_angle=st.session_state.cam_v_angle,
        cam_h_angle=st.session_state.cam_h_angle,
        cam_distance=st.session_state.cam_distance,
        max_view_distance=st.session_state.max_view_distance,
        key=viewer_key
    )


def render_download_buttons():
    "Render the download buttons for the CAD files"
    stl_file = st.session_state.current_stl_path
    if not stl_file or not Path(stl_file).exists():
        return
        
    step_file = stl_file.replace(".stl", ".step")

    left_col, right_col = st.columns([1, 1])
    with left_col:
        if Path(step_file).exists():
            with open(step_file, "rb") as file:
                st.download_button(
                    label="Download STEP",
                    data=file,
                    file_name=Path(step_file).name,
                    mime="application/octet-stream"
                )
    with right_col:
        if Path(stl_file).exists():
            with open(stl_file, "rb") as file:
                st.download_button(
                    label="Download STL",
                    data=file,
                    file_name=Path(stl_file).name,
                    mime="application/octet-stream"
                )


def inject_design_system():
    """Install a polished product-grade visual design system for the Streamlit interface."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

        :root {
            --bg: #050816;
            --panel: rgba(12, 18, 32, 0.78);
            --panel-strong: rgba(15, 23, 42, 0.94);
            --panel-soft: rgba(30, 41, 59, 0.58);
            --stroke: rgba(148, 163, 184, 0.18);
            --stroke-strong: rgba(125, 211, 252, 0.38);
            --text: #f8fafc;
            --muted: #94a3b8;
            --accent: #38bdf8;
            --accent-2: #8b5cf6;
            --success: #34d399;
            --shadow: 0 28px 90px rgba(0, 0, 0, 0.42);
        }

        html, body, .stApp, [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 12% 8%, rgba(56, 189, 248, 0.20), transparent 28rem),
                radial-gradient(circle at 84% 0%, rgba(139, 92, 246, 0.20), transparent 30rem),
                linear-gradient(135deg, #030712 0%, var(--bg) 46%, #0b1020 100%) !important;
            color: var(--text) !important;
            font-family: 'Inter', sans-serif !important;
        }

        header, footer, [data-testid="stToolbar"] { visibility: hidden; height: 0; }
        .block-container { padding: 1.25rem 2rem 2rem !important; max-width: 1560px !important; }
        * { font-family: 'Inter', sans-serif !important; }
        code, pre { font-family: 'JetBrains Mono', monospace !important; }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(2, 6, 23, 0.98), rgba(15, 23, 42, 0.96)) !important;
            border-right: 1px solid var(--stroke);
            box-shadow: 20px 0 80px rgba(0,0,0,.24);
        }
        [data-testid="stSidebar"] h1 { letter-spacing: -0.04em; font-weight: 850; }

        .app-shell {
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 30px;
            padding: 1.2rem;
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.58), rgba(2, 6, 23, 0.28));
            box-shadow: var(--shadow);
            backdrop-filter: blur(20px);
        }
        .hero-card {
            position: relative;
            overflow: hidden;
            padding: 1.55rem 1.7rem;
            border: 1px solid rgba(125, 211, 252, 0.24);
            border-radius: 28px;
            background:
                linear-gradient(135deg, rgba(56, 189, 248, 0.18), rgba(139, 92, 246, 0.13)),
                rgba(15, 23, 42, 0.62);
            box-shadow: 0 22px 80px rgba(2, 6, 23, 0.36);
            margin-bottom: 1rem;
        }
        .hero-card:after {
            content: "";
            position: absolute;
            inset: -40% -20% auto auto;
            width: 28rem;
            height: 28rem;
            background: radial-gradient(circle, rgba(56,189,248,.26), transparent 62%);
            pointer-events: none;
        }
        .eyebrow { color: #bae6fd; font-size: .76rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
        .hero-title { margin: .25rem 0 .35rem; font-size: clamp(2.1rem, 5vw, 4.3rem); line-height: .9; letter-spacing: -.09em; font-weight: 900; }
        .hero-copy { color: #cbd5e1; max-width: 820px; font-size: 1.02rem; margin: 0; }
        .hero-metrics { display: flex; flex-wrap: wrap; gap: .65rem; margin-top: 1.15rem; }
        .metric-pill {
            border: 1px solid rgba(148,163,184,.18);
            border-radius: 999px;
            padding: .5rem .78rem;
            background: rgba(2,6,23,.32);
            color: #dbeafe;
            font-size: .82rem;
            font-weight: 700;
        }
        .status-badge {
            display:inline-flex; align-items:center; gap:.45rem; padding:.42rem .7rem;
            border-radius:999px; background:rgba(52,211,153,.11); color:#a7f3d0;
            border:1px solid rgba(52,211,153,.28); font-weight:800; font-size:.78rem;
        }
        .status-dot { width:.48rem; height:.48rem; border-radius:999px; background:var(--success); box-shadow:0 0 0 .28rem rgba(52,211,153,.12); }

        .section-title { margin: .25rem 0 .85rem; font-size: 1.02rem; font-weight: 850; letter-spacing: -.03em; color: #f8fafc; }
        .section-subtitle { color: var(--muted); margin-top: -.55rem; margin-bottom: 1rem; font-size: .88rem; }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--stroke) !important;
            background: var(--panel) !important;
            border-radius: 24px !important;
            box-shadow: 0 18px 60px rgba(0,0,0,.22) !important;
            backdrop-filter: blur(18px);
        }
        div[data-testid="column"] { background: transparent !important; border: 0 !important; padding: .35rem !important; box-shadow: none !important; }

        div[data-testid="stChatMessage"] {
            border: 1px solid rgba(148,163,184,.14);
            border-radius: 22px;
            background: rgba(15,23,42,.52);
            padding: .35rem .5rem;
            margin-bottom: .75rem;
        }

        label[data-testid="stWidgetLabel"] p, .stSlider label, p, li { color: #cbd5e1 !important; }
        h1, h2, h3 { color: #f8fafc !important; }
        div[data-baseweb="input"], div[data-baseweb="select"] > div, textarea, input {
            background: rgba(2, 6, 23, 0.44) !important;
            border: 1px solid rgba(148, 163, 184, 0.20) !important;
            border-radius: 14px !important;
            color: #f8fafc !important;
            box-shadow: none !important;
        }
        div[data-baseweb="input"]:focus-within, div[data-baseweb="select"]:focus-within > div {
            border-color: var(--stroke-strong) !important;
            box-shadow: 0 0 0 4px rgba(56, 189, 248, 0.10) !important;
        }

        div.stButton > button, div.stDownloadButton > button {
            border-radius: 14px !important;
            border: 1px solid rgba(125, 211, 252, 0.26) !important;
            background: linear-gradient(135deg, rgba(56,189,248,.16), rgba(139,92,246,.13)) !important;
            color: #e0f2fe !important;
            font-weight: 800 !important;
            transition: transform .15s ease, border-color .15s ease, background .15s ease !important;
        }
        div.stButton > button:hover, div.stDownloadButton > button:hover {
            transform: translateY(-1px);
            border-color: rgba(125, 211, 252, 0.55) !important;
            background: linear-gradient(135deg, rgba(56,189,248,.28), rgba(139,92,246,.22)) !important;
        }
        .st-key-send_btn button {
            border-radius: 999px !important;
            background: linear-gradient(135deg, #38bdf8, #818cf8) !important;
            color: #020617 !important;
            box-shadow: 0 12px 30px rgba(56,189,248,.28) !important;
        }
        .st-key-reset_session_btn button, .st-key-drawing_uploader button { border-radius: 999px !important; }

        [data-testid="stExpander"] {
            border: 1px solid rgba(148,163,184,.16) !important;
            border-radius: 18px !important;
            background: rgba(2,6,23,.34) !important;
        }
        [data-testid="stStatusWidget"] { border-radius: 18px !important; }
        .stAlert { border-radius: 18px !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    """Render the product-facing hero panel."""
    model_label = st.session_state.get("selected_model", "Default Gemini")
    st.markdown(
        f"""
        <section class="hero-card">
          <div class="eyebrow">Mechanical Design Agent</div>
          <h1 class="hero-title">MEDA</h1>
          <p class="hero-copy">A sleek multi-agent CAD studio for turning sketches and prompts into verified parametric geometry with live iteration history, sandbox execution, and export-ready manufacturing files.</p>
          <div class="hero-metrics">
            <span class="status-badge"><span class="status-dot"></span>{model_label} active</span>
            <span class="metric-pill">Parametric CAD</span>
            <span class="metric-pill">Visual critique loop</span>
            <span class="metric-pill">STEP + STL export</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

def main():
    "Main function to run the Streamlit app"
    st.set_page_config(
        page_title="MEDA | Mechanical Design Agent",
        page_icon="🛠️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    initialize_session_state()
    render_llm_config_sidebar()

    inject_design_system()
    render_hero()

    if not st.session_state.config_created or not st.session_state.llm_config:
        st.error("⚠️ GEMINI_API_KEY environment variable is not configured. Please set the variable and reload.")
        return

    left_col, right_col = st.columns([1.3, 1])

    with left_col:
        render_chat_workspace()
        
    with right_col:
        render_right_column()

    # Trigger design generation loop if requested
    if st.session_state.get("is_generating", False) and st.session_state.get("active_prompt"):
        prompt = st.session_state.active_prompt
        st.session_state.active_prompt = None
        
        try:
            st.session_state.log_history = ""
            st.session_state.adk_events = []
            
            if not st.session_state.core:
                session_dir = f"NewCADs/run_{int(time.time())}"
                config = st.session_state.get("llm_config", {})
                model = config.get("model", "gemini-3.5-flash")
                key = config.get("api_key")
                st.session_state.core = ReasoningCADCore(
                    working_dir=session_dir,
                    model_name=model,
                    api_key=key
                )
            
            def streamlit_log_callback(msg: str):
                st.session_state.log_history += msg + "\n"
                if "active_log_placeholder" in st.session_state and st.session_state.active_log_placeholder:
                    st.session_state.active_log_placeholder.code(st.session_state.log_history, language="log")
            
            def streamlit_event_callback(ev_dict: dict):
                st.session_state.adk_events.append(ev_dict)
                if "active_status_placeholder" in st.session_state and st.session_state.active_status_placeholder:
                    with st.session_state.active_status_placeholder.container():
                        render_events_status(st.session_state.adk_events)
            
            st.session_state.core.log_callback = streamlit_log_callback
            st.session_state.core.event_callback = streamlit_event_callback
            
            keep_canvas = bool(st.session_state.generated_py_file)
            
            result = st.session_state.core.run_design_loop(
                prompt=prompt,
                constraints={},
                image_path=st.session_state.current_image_path,
                session_id=st.session_state.session_id,
                keep_canvas=keep_canvas,
                fast_mode=st.session_state.get("fast_mode", False),
                num_candidates=st.session_state.get("num_candidates", 1)
            )
            
            st.session_state.session_id = result.get("session_id")
            
            if result["success"]:
                st.session_state.generated_py_file = f"{st.session_state.core.sandbox.working_dir}/001.py"
                st.session_state.current_stl_path = f"{st.session_state.core.sandbox.working_dir}/001.stl"
                if "color" in result and result["color"]:
                    st.session_state.color = result["color"]
                st.session_state.stl_timestamp = time.time()
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Successfully generated/refined CAD model! You can preview the 3D model on the right panel. \n\n*Iterations: {result['iterations']}*",
                    "logs": st.session_state.log_history,
                    "adk_events": list(st.session_state.adk_events)
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Design loop completed with failures. Inspect the trace timeline below for details.",
                    "logs": st.session_state.log_history,
                    "adk_events": list(st.session_state.adk_events)
                })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Error generating CAD model: {str(e)}",
                "logs": st.session_state.log_history,
                "adk_events": list(st.session_state.adk_events)
            })
        finally:
            st.session_state.is_generating = False
            st.rerun()


if __name__ == "__main__":
    main()
