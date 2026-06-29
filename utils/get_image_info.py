"Get the latest png file from a directory"
import base64
import os
from mimetypes import guess_type
from pathlib import Path
from typing import Optional
import datetime
import csv

from openai import AzureOpenAI
from google import genai
from google.genai import types


# Function to encode a local image into data URL
def local_image_to_data_url(image_path):
    "Conver the local image to data url"
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(
            image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

# # Example usage
# image_path = '<path_to_image>'
# data_url = local_image_to_data_url(image_path)
# print("Data URL:", data_url)


def get_latest_png(directory="NewCADs") -> Optional[str]:
    """
    Get the absolute path of the most recently created PNG file in the specified directory.

    Args:
        directory (str): Path to the directory containing PNG files

    Returns:
        Optional[str]: Formatted absolute image path string in the format '<img {absolute_path}>' if PNG found,
                      None if no PNG files exist in the directory
    """
    try:
        # Convert directory to absolute Path object
        dir_path = Path(directory).resolve()
        # Get all PNG files in the directory
        png_files = list(dir_path.glob('*.png'))

        if not png_files:
            return None

        # Get the most recent file based on creation time
        latest_png = max(png_files, key=lambda x: x.stat().st_ctime)

        # Convert to absolute path and format as requested
        absolute_path = latest_png.absolute()
        print(absolute_path)
        return absolute_path

    except FileNotFoundError as e:
        print(f"Directory not found: {e}")
        return None
    except PermissionError as e:
        print(f"Permission denied: {e}")
        return None
    except OSError as e:
        print(f"OS error: {e}")
        return None

def get_image_info(prompt,cad_working_dir="NewCADs"):
    """
    Get image information from the API and track token usage.

    Args:
        prompt (str): The prompt to compare the image against
    Returns:
        str: The API response content
    """
    # Dynamic offscreen screenshot generation from the latest STL file
    latest_stl = None
    try:
        dir_path = Path(cad_working_dir).resolve()
        stl_files = list(dir_path.glob('*.stl'))
        if stl_files:
            latest_stl = max(stl_files, key=lambda x: x.stat().st_ctime)
    except Exception as e:
        print(f"[WARNING] Failed to locate latest STL file: {e}", flush=True)

    if latest_stl:
        png_path = latest_stl.with_suffix('.png')
        print(f"[LOG] Dynamically generating headless screenshot from {latest_stl.name} to {png_path.name}...", flush=True)
        try:
            import subprocess
            import sys
            script_path = Path(__file__).parent / "capture_screenshot.py"
            result = subprocess.run(
                [sys.executable, str(script_path), str(latest_stl), str(png_path)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("[LOG] Headless screenshot generated successfully via subprocess!", flush=True)
            else:
                print(f"[WARNING] Headless screenshot subprocess failed: {result.stderr.strip()}", flush=True)
        except Exception as e:
            print(f"[WARNING] Headless screenshot invocation failed: {e}", flush=True)

    image_path = get_latest_png(cad_working_dir)
    if not image_path:
        raise ValueError("No PNG file found in the specified directory")
        
    system_instruction = """
You are a helpful assistant that analyzes the isometric image of a CAD model. Follow these instructions precisely:

1. **Always begin** with a brief and objective description of the CAD model as seen in the image.
2. **Compare** your description with the provided prompt.
3. **Do not evaluate dimensions.** Focus only on the visual and structural appearance — not on measurements or annotations.
4. Your goal is to evaluate whether the visual appearance of the model matches the prompt.

### Evaluation:

- If the model's appearance **matches** the prompt based on your description:  
  ➤ Respond with **"SUCCESS and TERMINATE"**.

- If the model's appearance **does not match** the prompt:  
  ➤ Respond with **"FAILURE"**, followed by:
  - Corrected CAD generation steps in bullet points, using a parametric and sequential approach ensuring proper placement of CAD features."""

    if os.getenv("GEMINI_API_KEY"):
        # Use Google Gemini API
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/png"
                ),
                prompt
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3,
            )
        )
        completion_tokens = response.usage_metadata.candidates_token_count
        prompt_tokens = response.usage_metadata.prompt_token_count
        total_tokens = response.usage_metadata.total_token_count
        total_cost = (prompt_tokens / 1_000_000) * 0.075 + (completion_tokens / 1_000_000) * 0.30
        
        print(f"Completion tokens: {completion_tokens}")
        print(f"Prompt tokens: {prompt_tokens}")
        print(f"Total tokens: {total_tokens}")
        print(f"Total cost: {total_cost}")
        
        save_token_usage(prompt, completion_tokens, prompt_tokens, total_tokens, total_cost)
        return response.text

    elif os.getenv("AZURE_API_KEY"):
        # Use Azure OpenAI
        image_url = local_image_to_data_url(image_path)
        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_BASE"),
            api_key=os.getenv("AZURE_API_KEY"),
            api_version="2024-08-01-preview",
        )
        response = client.chat.completions.create(
            model="GPT-4o-0806",
            seed=43,
            temperature=0.3,
            messages=[
                {
                    "role": "system",
                    "content": system_instruction
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{prompt}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url 
                            }
                        }
                    ]
                }
            ],
        )
        completion_tokens = response.usage.completion_tokens
        prompt_tokens = response.usage.prompt_tokens
        cached_tokens = response.usage.prompt_tokens_details.cached_tokens
        total_tokens = response.usage.total_tokens
        total_cost = (prompt_tokens / 1_000_000)*2.5+ (cached_tokens / 1_000_000) * 1.25+ (completion_tokens/ 1_000_000) * 10 
        
        print(f"Completion tokens: {completion_tokens}")
        print(f"Prompt tokens: {prompt_tokens}")
        print(f"Total tokens: {total_tokens}")
        print(f"Total cost: {total_cost}")
        
        save_token_usage(prompt, completion_tokens, prompt_tokens, total_tokens, total_cost)
        return response.choices[0].message.content
    else:
        raise ValueError("Neither GEMINI_API_KEY nor AZURE_API_KEY is set in the environment variables.")

def save_token_usage(prompt, completion_tokens, prompt_tokens, total_tokens,total_cost):
    """
    Save token usage data to a CSV file.
    
    Args:
        prompt (str): The prompt used
        completion_tokens (int): Number of completion tokens used
        prompt_tokens (int): Number of prompt tokens used
        total_tokens (int): Total number of tokens used
    """
    csv_path = "tests/results/token_usage_log_seed_50.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(csv_path)
    
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Open file in append mode
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write headers if file is new
        if not file_exists:
            writer.writerow(["Timestamp", "Prompt", "Completion Tokens", "Prompt Tokens", "Total Tokens","Total Cost"])
        
        # Write the data row
        writer.writerow([timestamp, prompt, completion_tokens, prompt_tokens, total_tokens,total_cost])