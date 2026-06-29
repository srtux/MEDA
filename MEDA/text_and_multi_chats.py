"""
This module facilitates agentic chats for collaborative design problem solving.
This is specifically created for Streamlit app use.
"""
import logging
import sys
import traceback

# Setup standard logging to output everything to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("MEDA.chats")

from autogen import GroupChat, GroupChatManager
from autogen.agentchat.contrib.capabilities.vision_capability \
    import VisionCapability
from utils.path_finder import file_path_finder
from utils.image_path_changer import update_image_path


def multimodal_designers_chat(agents, config, design_problem: str):
    """
    Creates a group chat environment for collaborative design problem solving.
    """
    print(f"\n[LOG] Starting multimodal_designers_chat with prompt: {design_problem}", flush=True)
    print(f"[LOG] Configuration passed: {config}\n", flush=True)
    try:
        # Replace image file paths with <img image_path>
        agents_list = agents
        design_problem = update_image_path(design_problem)
        
        # Define speaker transition graph:
        # 0: user, 1: design_expert, 2: cad_coder, 3: executor, 4: reviewer, 5: cad_image_reviewer, 6: cad_data_reviewer
        graph_dict = {
            agents_list[0]: [agents_list[6], agents_list[1]],
            agents_list[6]: [agents_list[1]],
            agents_list[1]: [agents_list[2]],
            agents_list[2]: [agents_list[3]],
            agents_list[3]: [agents_list[4]],
            agents_list[4]: [agents_list[1], agents_list[2], agents_list[5]],
            agents_list[5]: [agents_list[1]]
        }

        print("[LOG] Setting up Multimodal GroupChat with transition graph...", flush=True)
        groupchat = GroupChat(
            agents=agents_list,
            messages=[],
            max_round=50,
            speaker_selection_method="auto",
            allow_repeat_speaker=None,
            func_call_filter=True,
            select_speaker_auto_verbose=True,
            send_introductions=True,
            allowed_or_disallowed_speaker_transitions=graph_dict,
            speaker_transitions_type="allowed"
        )
        vision_capability = VisionCapability(lmm_config={"config_list": [config]})
        group_chat_manager = GroupChatManager(
            groupchat=groupchat, llm_config={"config_list": [config]})
        vision_capability.add_to_agent(group_chat_manager)

        print("[LOG] Initiating group chat conversation...", flush=True)
        rst = agents_list[0].initiate_chat(
            group_chat_manager,
            message=design_problem,
        )
        print("[LOG] Group chat completed successfully.", flush=True)
        output = rst.chat_history
        return file_path_finder(output)
    except Exception as e:
        print(f"\n[ERROR] Exception occurred in multimodal_designers_chat: {str(e)}", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise e


def designers_chat(agents, config, design_problem: str):
    """
    Creates a group chat environment for collaborative design problem solving.
    """
    print(f"\n[LOG] Starting designers_chat with prompt: {design_problem}", flush=True)
    print(f"[LOG] Configuration passed: {config}\n", flush=True)
    try:
        agents_list = agents
        
        # Define speaker transition graph:
        # 0: user, 1: design_expert, 2: cad_coder, 3: executor, 4: reviewer, 5: cad_image_reviewer
        graph_dict = {
            agents_list[0]: [agents_list[1]],
            agents_list[1]: [agents_list[2]],
            agents_list[2]: [agents_list[3]],
            agents_list[3]: [agents_list[4]],
            agents_list[4]: [agents_list[1], agents_list[2], agents_list[5]],
            agents_list[5]: [agents_list[1]]
        }

        print("[LOG] Setting up Text GroupChat with transition graph...", flush=True)
        groupchat = GroupChat(
            agents=agents_list,
            messages=[],
            max_round=50,
            speaker_selection_method="auto",
            allow_repeat_speaker=None,
            func_call_filter=True,
            select_speaker_auto_verbose=True,
            send_introductions=True,
            allowed_or_disallowed_speaker_transitions=graph_dict,
            speaker_transitions_type="allowed"
        )
        manager = GroupChatManager(groupchat=groupchat, llm_config={"config_list": [config]})

        # Start chatting with the designer as this is the user proxy agent.
        print("[LOG] Initiating group chat conversation...", flush=True)
        response = agents_list[0].initiate_chat(
            manager,
            message=design_problem,
        )
        print("[LOG] Group chat completed successfully.", flush=True)
        output = response.chat_history
        return file_path_finder(output)
    except Exception as e:
        print(f"\n[ERROR] Exception occurred in designers_chat: {str(e)}", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise e