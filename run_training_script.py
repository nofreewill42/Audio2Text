import subprocess
import time

def run_script():
    # Replace 'your_training_script.py' with the path to your training script
    command = ["python3", "train.py"]
    # Run the script
    process = subprocess.Popen(command)

    # Wait for the script to complete
    try:
        stdout, stderr = process.communicate()
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

    return process.returncode

# Main loop to keep the script running
while True:
    result = run_script()

    # # Check if the script exited with a non-zero (error) status
    # if result != 0:
    #     print("Script stopped unexpectedly. Restarting...")
    #     # Optionally, you could include a delay before restarting
    #     # time.sleep(10)
    # else:
    #     # If the script completed successfully, no need to restart
    #     print("Script completed successfully. Exiting.")
    #     break
