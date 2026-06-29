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
from streamlit_utils.prompt_builder import PromptBuilder
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
        st.session_state.selected_model = None
    if 'llm_config' not in st.session_state:
        st.session_state.llm_config = None
    if 'config_created' not in st.session_state:
        st.session_state.config_created = False
    default_state = StateManager.get_default_state()
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.stl_timestamp is None:
        st.session_state.stl_timestamp = time.time()
    if 'log_history' not in st.session_state:
        st.session_state.log_history = ""


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


def render_controls():
    "Render the main controls for the Streamlit app"

    text_prompt = st.text_input("Let's design",
                                value=st.session_state.prompt,
                                placeholder="Enter a text prompt here",
                                key="input_prompt")
    uploaded_file = None
    if st.session_state.selected_model != "Text LLM":
        uploaded_file = st.file_uploader(
            "Upload an engineering drawing image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        if not st.session_state.current_image_path or not Path(st.session_state.current_image_path).exists():
            st.session_state.current_image_path = FileHandler.save_uploaded_file(uploaded_file)
        if st.session_state.current_image_path and Path(st.session_state.current_image_path).exists():
            st.image(st.session_state.current_image_path, caption="Uploaded Image",
                     use_container_width=True)

    if st.button("Generate CAD Model"):
        if text_prompt:
            with st.spinner("Generating CAD model..."):
                try:
                    st.session_state.log_history = ""
                    
                    def streamlit_log_callback(msg: str):
                        st.session_state.log_history += msg + "\n"
                        if "log_placeholder" in st.session_state and st.session_state.log_placeholder:
                            st.session_state.log_placeholder.code(st.session_state.log_history, language="log")

                    # Live per-iteration render placeholder so the user watches
                    # the model improve each turn.
                    render_placeholder = st.empty()

                    def streamlit_render_callback(iter_n: int, png_path: str):
                        try:
                            render_placeholder.image(
                                png_path,
                                caption=f"Iteration {iter_n} — current model",
                                use_container_width=True,
                            )
                        except Exception:
                            pass

                    session_dir = f"NewCADs/run_{int(time.time())}"
                    config = st.session_state.get("llm_config", {})
                    model = config.get("model", "gemini-3.5-flash")
                    key = config.get("api_key")
                    
                    core = ReasoningCADCore(
                        working_dir=session_dir,
                        model_name=model,
                        api_key=key
                    )
                    core.log_callback = streamlit_log_callback
                    core.render_callback = streamlit_render_callback

                    result = core.run_design_loop(
                        prompt=text_prompt,
                        constraints={},
                        image_path=st.session_state.current_image_path
                    )
                    if result["success"]:
                        st.session_state.generated_py_file = f"{session_dir}/001.py"
                        st.session_state.current_stl_path = f"{session_dir}/001.stl"
                        if "color" in result and result["color"]:
                            st.session_state.color = result["color"]
                        st.rerun()
                    else:
                        st.warning("Design loop completed with failures. Inspect the trace log below for details.")
                except Exception as e:
                    st.error(f"Error generating CAD model: {str(e)}")


def visualization_controls():
    "Render the visualization controls for the CAD model"
    # st.subheader("Visualization Controls")

    controls = {
        'color': st.color_picker("Pick a color", value=st.session_state.color),
        'material': st.selectbox("Select a material",
                                 ["material", "flat", "wireframe"],
                                 index=["material", "flat", "wireframe"].index(st.session_state.material)),
        'auto_rotate': st.toggle("Auto rotation", value=st.session_state.auto_rotate),
        'opacity': st.slider("Opacity", min_value=0.0, max_value=1.0, value=st.session_state.opacity),
        'height': st.slider("Height", min_value=50, max_value=1000, value=st.session_state.height)
    }

    # st.subheader("Camera Controls")
    # camera_controls = {
    #     'cam_v_angle': st.number_input("Camera Vertical Angle", value=st.session_state.cam_v_angle),
    #     'cam_h_angle': st.number_input("Camera Horizontal Angle", value=st.session_state.cam_h_angle),
    #     'cam_distance': st.number_input("Camera Distance", value=st.session_state.cam_distance),
    #     'max_view_distance': st.number_input("Max view distance", min_value=1, value=st.session_state.max_view_distance)
    # }

    for key, value in {**controls}.items():
        st.session_state[key] = value


def render_stl_viewer():
    "Render the STL viewer"
    viewer_key = f'stl_viewer_{st.session_state.stl_timestamp}'
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
    step_file = stl_file.replace(".stl", ".step")

    left_col, right_col = st.columns([4, 1])
    with left_col:
        with open(step_file, "rb") as file:
            st.download_button(
                label="Download CAD STEP",
                data=file,
                file_name=Path(step_file).name,
                mime="application/octet-stream"
            )
    with right_col:
        with open(stl_file, "rb") as file:
            st.download_button(
                label="Download CAD STL",
                data=file,
                file_name=Path(stl_file).name,
                mime="application/octet-stream"
            )


def render_example_prompts():
    "Render example prompts for the user"
    st.subheader("✨ Example prompts")
    examples = [
        "A box with a through hole in the center.",
        "Create a pipe of outer diameter 50mm and inside diameter 40mm.",
        "Create a circular plate of radius 2mm and thickness 0.125mm with four holes of radius 0.25mm patterned at distance of 1.5mm from the centre along the axes."
    ]
    for example in examples:
        if st.button(example, key=f"example_{example}"):
            st.session_state.prompt = example
            st.rerun()


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
    st.set_page_config(page_title="MEDA · LLM to CAD", page_icon="🛠️", layout="wide")
    initialize_session_state()
    inject_design_system()

    render_llm_config_sidebar()
    render_hero()

    left_col, middle_col, right_col = st.columns([0.85, 2, 0.65])

    if not st.session_state.config_created or not st.session_state.llm_config:
        st.info("👈 Please select and configure your LLM settings in the sidebar to get started.")
        return

    with left_col:
        render_controls()
        visualization_controls()
    with middle_col:
        render_stl_viewer()
        render_download_buttons()
        render_example_prompts()
        
    with right_col:
        if st.session_state.generated_py_file:
            render_parameter_controls(st.session_state.generated_py_file)

    # Full-width persistent log expander at the bottom of the page
    st.write("---")
    log_expander = st.expander("Real-time Agent Log Trace", expanded=True)
    st.session_state.log_placeholder = log_expander.empty()
    if st.session_state.log_history:
        st.session_state.log_placeholder.code(st.session_state.log_history, language="log")


if __name__ == "__main__":
    main()
