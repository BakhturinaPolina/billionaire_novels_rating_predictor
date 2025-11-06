import subprocess
import sys
import threading
import time
from datetime import datetime

# Path and name to the script you are trying to start
file_path = "./bertopic_plus_octis.py"
restart_timer = 2


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def stream_output(stream, prefix=''):
    for line in iter(stream.readline, ''):
        timestamp = get_timestamp()
        print(f"[{timestamp}] {prefix}{line.strip()}")


def run_and_monitor(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
                               universal_newlines=True)

    stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, ""))
    stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, ""))

    stdout_thread.start()
    stderr_thread.start()

    exit_code = process.wait()

    stdout_thread.join()
    stderr_thread.join()

    return exit_code


def start_script():
    try:
        # Make sure 'python' command is available
        command = ["python3", file_path]
        timestamp = get_timestamp()
        print(f"[{timestamp}] Starting script: {file_path}")
        exit_code = run_and_monitor(command)

        timestamp = get_timestamp()
        if exit_code == 0:
            print(f"[{timestamp}] Script completed successfully.")
        else:
            print(f"[{timestamp}] Script exited with code {exit_code}.")
            handle_crash()
    except Exception as e:
        # Script crashed, lets restart it!
        timestamp = get_timestamp()
        print(f"[{timestamp}] An exception occurred: {e}")
        handle_crash()


def handle_crash():
    timestamp = get_timestamp()
    print(f"[{timestamp}] Restarting script in {restart_timer} seconds...")
    time.sleep(restart_timer)  # Restarts the script after specified seconds
    start_script()


if __name__ == "__main__":
    start_script()