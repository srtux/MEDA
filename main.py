"""Terminal app for mechanical design agent using Google ADK."""
import os
import time
from pathlib import Path
from dotenv import load_dotenv

from core.reasoning_core import ReasoningCADCore

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 50)
    print(f"{title:^50}")
    print("=" * 50 + "\n")


def main():
    """Main terminal loop."""
    load_dotenv()
    
    clear_screen()
    print_header("MEDA: CAD Agent CLI (Google ADK)")
    
    # 1. Setup API Key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY environment variable not found.")
        api_key = input("Please enter your Gemini API Key: ").strip()
        if not api_key:
            print("Error: Gemini API Key is required to run the agent. Exiting.")
            return
        os.environ["GEMINI_API_KEY"] = api_key

    # 2. Select Model
    model_name = "gemini-3.5-flash"

    while True:
        print_header("New CAD Design Request")
        
        # 3. Get design prompt
        prompt = input("Let's design (Enter your CAD prompt, or 'q' to quit): ").strip()
        if not prompt:
            continue
        if prompt.lower() in ['q', 'quit', 'exit']:
            print("Exiting...")
            break
            
        # 4. Get optional image path
        image_path = input("Enter path to engineering drawing image (optional, press Enter to skip): ").strip()
        if image_path:
            img_path = Path(image_path)
            if not img_path.exists():
                print(f"Warning: Image file not found at '{image_path}'. Proceeding without image.")
                image_path = None
        else:
            image_path = None
            
        # 5. Run the design loop
        session_dir = f"NewCADs/run_{int(time.time())}"
        print(f"\nInitializing MEDA design session in: {session_dir}")
        print("Starting design loop...\n")
        
        try:
            core = ReasoningCADCore(
                working_dir=session_dir,
                model_name=model_name,
                api_key=api_key
            )
            
            result = core.run_design_loop(
                prompt=prompt,
                constraints={},
                image_path=image_path
            )
            
            print_header("Design Run Summary")
            if result["success"]:
                print("SUCCESS: CAD model generated successfully!")
                print(f"Generated Python script: {session_dir}/001.py")
                print(f"Generated STL file:      {session_dir}/001.stl")
                print(f"Generated STEP file:     {session_dir}/001.step")
                print(f"Suggested Color:         {result.get('color')}")
            else:
                print("FAILED: Design loop completed but failed to satisfy all constraints.")
                if result.get("failed_constraints"):
                    print("Violations:")
                    for violation in result["failed_constraints"]:
                        print(f" - {violation}")
                        
        except Exception as e:
            print(f"\nError encountered during design process: {str(e)}")
            
        input("\nPress Enter to start a new design or select another action...")
        clear_screen()

if __name__ == "__main__":
    main()
