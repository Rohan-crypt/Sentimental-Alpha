import subprocess
import sys
import os
import time

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def run_script(script_path, is_streamlit=False):
    python_exe = os.path.join("venv", "Scripts", "python.exe")
    streamlit_exe = os.path.join("venv", "Scripts", "streamlit.exe")
    
    if is_streamlit:
        cmd = [streamlit_exe, "run", script_path]
    else:
        cmd = [python_exe, script_path]
    
    print(f"\n[EXEC] Running: {' '.join(cmd)}...")
    try:
        # We use Popen for streamlit so it doesn't block the script if we want to do more, 
        # but for this simple orchestrator, wait() is fine.
        process = subprocess.Popen(cmd)
        process.wait()
    except KeyboardInterrupt:
        print("\n[STOP] Process interrupted by user.")
        process.terminate()

def main():
    while True:
        clear_screen()
        print("==============================================")
        print("   🚀 SENTIMENTAL-ALPHA: COMMAND CENTER   ")
        print("==============================================")
        print("1. 🔥 Train New Brain (main_app.py)")
        print("2. 📊 Run Backtest & Update Data (test_agent.py)")
        print("3. 🖥️  Launch Research Dashboard (dashboard.py)")
        print("4. ⚡ Quick Start: Backtest + Dashboard")
        print("5. ❌ Exit")
        print("==============================================")
        
        choice = input("\nSelect an option (1-5): ")
        
        if choice == '1':
            run_script("main_app.py")
            input("\nTraining complete. Press Enter to return to menu...")
        elif choice == '2':
            run_script("test_agent.py")
            input("\nBacktest complete. Results updated. Press Enter to return to menu...")
        elif choice == '3':
            run_script("dashboard.py", is_streamlit=True)
        elif choice == '4':
            print("\n[Step 1/2] Updating backtest results...")
            run_script("test_agent.py")
            print("\n[Step 2/2] Launching dashboard...")
            run_script("dashboard.py", is_streamlit=True)
        elif choice == '5':
            print("Shutting down Sentimental-Alpha. Happy Trading!")
            break
        else:
            print("Invalid selection. Try again.")
            time.sleep(1)

if __name__ == "__main__":
    main()
