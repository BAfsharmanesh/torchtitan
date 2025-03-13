import subprocess
import psutil
import signal
import time
import os

def run_with_timeout(command, timeout_seconds):
    try:
        # Start the process without waiting for it to complete
        process = subprocess.Popen(command)
        main_pid = process.pid
        
        # Wait for the specified timeout
        process.wait(timeout=timeout_seconds)
        print("-TIMEOUT- Process completed successfully within the timeout period")
    except subprocess.TimeoutExpired:
        print(f"-TIMEOUT- Process exceeded timeout of {timeout_seconds} seconds. Terminating all related processes...")
        
        # Find and kill all child processes
        parent = psutil.Process(main_pid)
        children = parent.children(recursive=True)
        
        # Kill children first
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        
        # Then kill the parent
        try:
            parent.kill()
        except psutil.NoSuchProcess:
            pass
            
        print("-TIMEOUT- Process was terminated due to timeout")

# # Example usage
# command = "python ./playground/test.py"
# run_with_timeout(command, timeout_seconds=15)  # 5 minute timeout