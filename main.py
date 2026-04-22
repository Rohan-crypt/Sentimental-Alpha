import subprocess
import sys
import os
import time
import signal

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def run_script(script_path, args=None, is_streamlit=False, is_fastapi=False):
    python_exe = os.path.join("venv", "Scripts", "python.exe")
    streamlit_exe = os.path.join("venv", "Scripts", "streamlit.exe")
    uvicorn_exe = os.path.join("venv", "Scripts", "uvicorn.exe")
    
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
        is_running = api_process and api_process.poll() is None
        api_status = "RUNNING" if is_running else "STOPPED"
        print(f"Inference Server: {api_status}")
        print("----------------------------------------------")
        print("1. Train Model")
        
        if is_running:
            print("2. STOP API Server")
        else:
            print("2. START API Server")
            
        print("3. Launch Dashboard")
        print("4. Automated System Startup")
        print("5. Performance Validation Report")
        print("6. Exit & Shutdown All")
        print("----------------------------------------------")
        
        choice = input("Select Option (1-6): ")
        
        if choice == '1':
            run_script("main_app.py")
            input("\nProcess complete. Press Enter...")
            
        elif choice == '2':
            if is_running:
                print("Shutting down API server safely...")
                api_process.terminate()
                api_process.wait()
                api_process = None
                print("Server stopped.")
                time.sleep(1)
            else:
                print("Starting API server...")
                api_process = run_script("api.py", is_fastapi=True)
                time.sleep(3)
            
        elif choice == '3':
            if not is_running:
                print("Starting required API server...")
                api_process = run_script("api.py", is_fastapi=True)
                time.sleep(4)
            run_script("dashboard.py", is_streamlit=True)
            
        elif choice == '4':
            if is_running:
                api_process.terminate()
                api_process.wait()
            print("Initializing backend...")
            api_process = run_script("api.py", is_fastapi=True)
            time.sleep(5)
            print("Initializing frontend...")
            run_script("dashboard.py", is_streamlit=True)
            
        elif choice == '5':
            target = input("Enter ticker for validation (e.g. AAPL): ") or "AAPL"
            run_script("validate_model.py", args=[target.upper()])
            input("\nReport generated. Press Enter...")
            
        elif choice == '6':
            if api_process and api_process.poll() is None:
                print("Graceful shutdown in progress...")
                api_process.terminate()
                api_process.wait()
            print("System offline. Goodbye.")
            break
        else:
            time.sleep(0.5)

if __name__ == "__main__":
    main()
