import subprocess
import sys
import os
import time
import signal

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_executables():
    """
    Auto-detects virtual environment paths.
    """
    possible_venvs = ["venv", ".venv"]
    for v in possible_venvs:
        py = os.path.join(v, "Scripts", "python.exe")
        st = os.path.join(v, "Scripts", "streamlit.exe")
        uv = os.path.join(v, "Scripts", "uvicorn.exe")
        if os.path.exists(py):
            return py, st, uv
    
    # Fallback to system python if venv not found (not recommended but safe)
    return sys.executable, "streamlit", "uvicorn"

def run_script(script_path, args=None, is_streamlit=False, is_fastapi=False):
    python_exe, streamlit_exe, uvicorn_exe = get_executables()
    
    if is_streamlit:
        cmd = [streamlit_exe, "run", script_path]
    elif is_fastapi:
        cmd = [uvicorn_exe, "api:app", "--host", "0.0.0.0", "--port", "8000"]
    else:
        cmd = [python_exe, script_path]
        if args:
            cmd.extend(args)
    
    try:
        if is_fastapi:
            return subprocess.Popen(cmd)
        else:
            print(f"Executing: {' '.join(cmd)}")
            process = subprocess.Popen(cmd)
            process.wait()
            return process
    except KeyboardInterrupt:
        return None
    except Exception as e:
        print(f"Error running {script_path}: {e}")
        return None

def main():
    api_process = None
    
    while True:
        clear_screen()
        print("==============================================")
        print("    SENTIMENTAL-ALPHA: AI COMMAND CENTER      ")
        print("==============================================")
        is_running = api_process and api_process.poll() is None
        api_status = "ONLINE" if is_running else "OFFLINE"
        print(f"Status: [INFERENCE ENGINE {api_status}]")
        print("----------------------------------------------")
        print("1. [TRAIN] Re-train AI Brain (Advanced 3-Layer)")
        print("2. [API] Start/Stop Inference Server")
        print("3. [UI] Launch Research Dashboard")
        print("4. [SYSTEM] Automated Full Startup")
        print("5. [VAL] Performance Validation Report")
        print("6. [VIS] Generate Research Analytics (Graphs)")
        print("7. [TEST] Run Test Agent (Manual Inference)")
        print("8. [EXIT] Shutdown All Services")
        print("----------------------------------------------")
        
        choice = input("Command >> ")
        
        if choice == '1':
            print("\nStarting Advanced Training Sequence...")
            run_script("main_app.py")
            input("\nTraining Complete. New 'nifty_alpha_brain' saved. Press Enter...")
            
        elif choice == '2':
            if is_running:
                print("Shutting down API server...")
                api_process.terminate()
                api_process.wait()
                api_process = None
                print("Server OFFLINE.")
                time.sleep(1)
            else:
                print("Starting API server...")
                api_process = run_script("api.py", is_fastapi=True)
                time.sleep(3)
            
        elif choice == '3':
            if not is_running:
                print("Dashboard requires API. Starting API first...")
                api_process = run_script("api.py", is_fastapi=True)
                time.sleep(4)
            run_script("dashboard.py", is_streamlit=True)
            
        elif choice == '4':
            if is_running:
                api_process.terminate()
                api_process.wait()
            print("Initializing AI Infrastructure...")
            api_process = run_script("api.py", is_fastapi=True)
            
            max_retries = 20
            ready = False
            for i in range(max_retries):
                try:
                    import requests
                    response = requests.get("http://127.0.0.1:8000/health", timeout=2)
                    if response.status_code == 200:
                        ready = True
                        break
                except Exception:
                    pass
                print(f"Waiting for Brain... ({i+1}/{max_retries})", end="\r")
                time.sleep(2)
            
            if ready:
                print("\nBrain Online. Launching Terminal...")
                run_script("dashboard.py", is_streamlit=True)
            else:
                print("\nError: API failed to start. Check terminal output.")
                input("Press Enter to return...")
            
        elif choice == '5':
            print("\n--- PERFORMANCE VALIDATION ---")
            target = input("Enter Ticker (default: AAPL): ").upper() or "AAPL"
            run_script("validate_model.py", args=[target])
            input("\nValidation Complete. Press Enter...")
            
        elif choice == '6':
            print("\n--- RESEARCH ANALYTICS (GRAPHS) ---")
            print("a. Backtest Results (Equity/Accuracy)")
            print("b. Training Convergence (Learning Progress)")
            print("c. Feature Dynamics (Explainability)")
            sub_choice = input("Select Type >> ").lower()
            
            if sub_choice == 'a':
                t = input("Ticker (default: AAPL): ").upper() or "AAPL"
                run_script("visualize_results.py", args=[t])
            elif sub_choice == 'b':
                run_script("visualize_training.py")
            elif sub_choice == 'c':
                run_script("visualize_features.py")
            input("\nAnalytics Generated. Press Enter...")

        elif choice == '7':
            print("\n--- TEST AGENT INFERENCE ---")
            target = input("Enter Ticker (default: RELIANCE.NS): ").upper() or "RELIANCE.NS"
            run_script("test_agent.py", args=[target])
            input("\nTest Run Complete. Press Enter...")

        elif choice == '8':
            if api_process and api_process.poll() is None:
                api_process.terminate()
                api_process.wait()
            print("All systems offline. Goodbye.")
            break
        else:
            time.sleep(0.5)

if __name__ == "__main__":
    main()
