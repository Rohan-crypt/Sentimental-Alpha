import subprocess
import sys
import os
import time

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def run_script(script_path, is_streamlit=False, is_fastapi=False):
    python_exe = os.path.join("venv", "Scripts", "python.exe")
    streamlit_exe = os.path.join("venv", "Scripts", "streamlit.exe")
    uvicorn_exe = os.path.join("venv", "Scripts", "uvicorn.exe")
    
    if is_streamlit:
        cmd = [streamlit_exe, "run", script_path]
    elif is_fastapi:
        cmd = [uvicorn_exe, "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
    else:
        cmd = [python_exe, script_path]
    
    try:
        if is_fastapi:
            return subprocess.Popen(cmd)
        else:
            process = subprocess.Popen(cmd)
            process.wait()
            return process
    except KeyboardInterrupt:
        return None

def main():
    api_process = None
    
    while True:
        clear_screen()
        print("----------------------------------------------")
        print("    SENTIMENTAL-ALPHA: COMMAND CENTER v2.0    ")
        print("----------------------------------------------")
        api_status = "RUNNING" if api_process and api_process.poll() is None else "STOPPED"
        print(f"Server Status: {api_status}")
        print("----------------------------------------------")
        print("1. Train Model (main_app.py)")
        print("2. Toggle API Server (api.py)")
        print("3. Launch Dashboard")
        print("4. Automated System Startup")
        print("5. Exit System")
        print("----------------------------------------------")
        
        choice = input("Select Option (1-5): ")
        
        if choice == '1':
            run_script("main_app.py")
            input("\nProcess complete. Press Enter...")
            
        elif choice == '2':
            if api_process and api_process.poll() is None:
                api_process.terminate()
                time.sleep(1)
            api_process = run_script("api.py", is_fastapi=True)
            time.sleep(2)
            
        elif choice == '3':
            if not api_process or api_process.poll() is not None:
                api_process = run_script("api.py", is_fastapi=True)
                time.sleep(3)
            run_script("dashboard.py", is_streamlit=True)
            
        elif choice == '4':
            if api_process and api_process.poll() is None:
                api_process.terminate()
            api_process = run_script("api.py", is_fastapi=True)
            time.sleep(4)
            run_script("dashboard.py", is_streamlit=True)
            
        elif choice == '5':
            if api_process and api_process.poll() is None:
                api_process.terminate()
            break
        else:
            time.sleep(1)

if __name__ == "__main__":
    main()
