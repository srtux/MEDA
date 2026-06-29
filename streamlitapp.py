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
    st.write("### Parametric controls")

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
    st.write("### 💬 CAD Chat")
    
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
    st.write("### 📐 3D Preview")
    
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
    """Install a lightweight visual design system for the Streamlit interface."""
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top left, #172554 0, #0f172a 32%, #020617 100%);
            color: #e5e7eb;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(15,23,42,.98), rgba(30,41,59,.98));
            border-right: 1px solid rgba(148,163,184,.22);
        }
        .hero-card {
            padding: 1.4rem 1.6rem;
            border: 1px solid rgba(148,163,184,.25);
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(59,130,246,.18), rgba(14,165,233,.08));
            box-shadow: 0 22px 80px rgba(2,6,23,.35);
            margin-bottom: 1rem;
        }
        .hero-card h1 { margin: 0; font-size: 3rem; letter-spacing: -0.08em; }
        .hero-card p { color: #bfdbfe; font-size: 1.05rem; margin-bottom: 0; }
        div.stButton > button, div.stDownloadButton > button {
            border-radius: 999px;
            border: 1px solid rgba(125,211,252,.5);
            background: linear-gradient(135deg, #38bdf8, #6366f1);
            color: white;
            font-weight: 700;
        }
        [data-testid="stExpander"] {
            border: 1px solid rgba(148,163,184,.25);
            border-radius: 18px;
            background: rgba(15,23,42,.72);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    """Render the product-facing hero panel."""
    st.markdown(
        """
        <section class="hero-card">
          <h1>MEDA</h1>
          <p>Multi-agent, executable CAD synthesis with parametric timelines, sandboxed B-Rep verification, and visual self-critique.</p>
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

    # Apply custom modern styling
    custom_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');

    /* Background and main container */
    html, body, [data-testid="stAppViewContainer"], .stApp, [data-testid="stHeader"], [data-testid="stToolbar"] {
        background-color: #080c14 !important;
        background: #080c14 !important;
    }

    header, footer {
        visibility: hidden;
    }

    /* Custom typography */
    h1, h2, h3, h4, h5, h6, p, li {
        font-family: 'Inter', sans-serif !important;
    }

    /* Style the main title */
    .title-banner {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 2rem;
        margin-bottom: 1.5rem;
        background: #0f172a;
        border-bottom: 1px solid #1e293b;
        border-radius: 12px;
    }
    .title-banner h1 {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        letter-spacing: -0.04em;
        margin: 0 !important;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .title-banner p {
        font-size: 0.95rem !important;
        color: #94a3b8 !important;
        margin: 0 !important;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: rgba(16, 185, 129, 0.1);
        color: #10b981;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 9999px;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    .status-dot {
        width: 6px;
        height: 6px;
        background-color: #10b981;
        border-radius: 50%;
    }

    /* Cards for columns */
    div[data-testid="column"] {
        background-color: #0f172a !important;
        border: 1px solid #1e293b !important;
        border-radius: 12px !important;
        padding: 24px !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    div[data-testid="column"]:hover {
        border-color: #334155 !important;
    }

    /* High-contrast widget labels */
    label[data-testid="stWidgetLabel"] p, .stSlider label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }

    /* Styled inputs */
    div[data-baseweb="input"], input, textarea, select {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        color: #f8fafc !important;
        padding: 6px !important;
        transition: border-color 0.15s ease !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #0ea5e9 !important;
        box-shadow: 0 0 0 1px #0ea5e9 !important;
    }

    /* Primary buttons (Generate CAD Model) */
    button[kind="primary"] {
        background-color: #0284c7 !important;
        background: #0284c7 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: background-color 0.15s ease, transform 0.1s ease !important;
        width: 100%;
        box-shadow: 0 4px 6px -1px rgba(2, 132, 199, 0.2), 0 2px 4px -2px rgba(2, 132, 199, 0.2) !important;
    }
    button[kind="primary"]:hover {
        background-color: #0369a1 !important;
        background: #0369a1 !important;
        transform: translateY(-1px);
    }
    button[kind="primary"]:active {
        transform: translateY(0);
    }

    /* Secondary / Outline buttons (Downloads, Example Prompts) */
    button[kind="secondary"], div[data-testid="stDownloadButton"] button {
        background-color: rgba(255, 255, 255, 0.02) !important;
        background: rgba(255, 255, 255, 0.02) !important;
        color: #cbd5e1 !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        transition: all 0.15s ease !important;
        width: 100%;
        text-align: center;
        box-shadow: none !important;
    }
    button[kind="secondary"]:hover, div[data-testid="stDownloadButton"] button:hover {
        background-color: rgba(2, 132, 199, 0.08) !important;
        background: rgba(2, 132, 199, 0.08) !important;
        border-color: #0284c7 !important;
        color: #38bdf8 !important;
    }

    /* Styled expander */
    .streamlit-expanderHeader {
        background-color: #0f172a !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px !important;
        padding: 12px 18px !important;
    }
    .streamlit-expanderContent {
        background-color: #080c14 !important;
        border-left: 1px solid #1e293b !important;
        border-right: 1px solid #1e293b !important;
        border-bottom: 1px solid #1e293b !important;
        border-bottom-left-radius: 8px !important;
        border-bottom-right-radius: 8px !important;
    }

    /* Log tracer code block styling */
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
        background-color: transparent !important;
        color: #e2e8f0 !important;
        border: none !important;
    }
    
    /* Center columns structure padding */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
    }

    /* 1. Scrollable chat messages container (first wrapper) */
    div[data-testid="column"]:first-child div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"]:first-of-type > div {
        height: calc(100vh - 150px) !important;
        max-height: calc(100vh - 150px) !important;
    }
    div[data-testid="column"]:first-child div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"]:first-of-type div[data-testid="stVerticalBlock"] {
        height: calc(100vh - 150px) !important;
        max-height: calc(100vh - 150px) !important;
    }

    /* 2. Unified Google Gemini-style prompt capsule container (second wrapper) */
    div[data-testid="column"]:first-child div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-testid="stTextInput"]) {
        background-color: rgba(15, 23, 42, 0.6) !important; /* slate-900 transparent */
        border: 1px solid #1e293b !important;
        border-radius: 36px !important; /* PERFECT CAPSULE PILL SHAPE! */
        padding: 8px 20px !important;
        margin-top: 10px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Make sure columns inside the capsule align vertically centered */
    div[data-testid="column"]:first-child div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-testid="stTextInput"]) div[data-testid="stHorizontalBlock"] {
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        gap: 8px !important;
    }
    
    /* Make text input inside prompt box borderless and transparent */
    div[data-testid="column"]:first-child div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-testid="stTextInput"]) div[data-testid="stTextInput"] input {
        border: none !important;
        background-color: transparent !important;
        padding-left: 8px !important;
        color: #e2e8f0 !important;
        box-shadow: none !important;
        font-size: 16px !important;
        height: 38px !important;
    }
    
    /* Plus (+) Upload button styling inside capsule */
    .st-key-drawing_uploader {
        padding: 0 !important;
        margin: 0 !important;
        width: 36px !important;
        min-width: 36px !important;
        max-width: 36px !important;
    }
    .st-key-drawing_uploader [data-testid="stFileUploader"] {
        width: 100% !important;
    }
    .st-key-drawing_uploader [data-testid="stWidgetLabel"] {
        display: none !important;
    }
    .st-key-drawing_uploader section[data-testid="stFileUploaderDropzone"] {
        border: none !important;
        background-color: transparent !important;
        min-height: 36px !important;
        height: 36px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
    }
    .st-key-drawing_uploader [data-testid="stFileUploaderDropzoneInstructions"] {
        display: none !important;
    }
    .st-key-drawing_uploader button {
        width: 36px !important;
        height: 36px !important;
        margin: 0 !important;
        background-color: transparent !important;
        border: none !important;
        color: transparent !important;
        font-size: 0px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
    }
    .st-key-drawing_uploader button::before {
        content: "📎" !important; /* paperclip emoji */
        font-size: 20px !important;
        display: inline-block !important;
        color: #94a3b8 !important;
        line-height: 36px !important;
    }
    .st-key-drawing_uploader button:hover::before {
        color: #38bdf8 !important;
    }
    
    /* Reset button styling inside capsule */
    .st-key-reset_session_btn {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        padding: 0 !important;
        margin: 0 !important;
        width: 36px !important;
        min-width: 36px !important;
        max-width: 36px !important;
    }
    .st-key-reset_session_btn button {
        height: 36px !important;
        border: none !important;
        background-color: transparent !important;
        color: #94a3b8 !important;
        font-size: 18px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 36px !important;
    }
    .st-key-reset_session_btn button:hover {
        background-color: rgba(2, 132, 199, 0.08) !important;
        color: #38bdf8 !important;
        border-radius: 50% !important; /* hover circle for reset icon! */
    }

    /* SOLID BLUE CIRCLE Send button styling on the right inside capsule */
    .st-key-send_btn {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        padding: 0 !important;
        margin: 0 !important;
        width: 36px !important;
        min-width: 36px !important;
        max-width: 36px !important;
    }
    .st-key-send_btn button {
        background-color: #38bdf8 !important; /* solid sky blue background! */
        color: #0f172a !important; /* dark slate arrow icon! */
        border: none !important;
        border-radius: 50% !important; /* PERFECT CIRCLE! */
        width: 36px !important;
        height: 36px !important;
        min-width: 36px !important;
        max-width: 36px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 2px 8px rgba(56, 189, 248, 0.3) !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    .st-key-send_btn button:hover {
        background-color: #0ea5e9 !important; /* hover slightly darker blue */
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.5) !important;
    }
    /* Set send icon size inside the button */
    .st-key-send_btn button p {
        font-size: 18px !important;
        margin: 0 !important;
        line-height: 36px !important;
        color: #0f172a !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    # Header section
    st.markdown("""
        <div class="title-banner">
            <h1>MEDA <span class="status-badge"><span class="status-dot"></span>Gemini 3.5 Active</span></h1>
            <p>Autonomous Mechanical Design & Parametric CAD Agents</p>
        </div>
    """, unsafe_allow_html=True)

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
