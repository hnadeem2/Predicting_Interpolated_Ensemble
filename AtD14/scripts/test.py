import subprocess

def run_bash_script(arg1, arg2):
    bash_script_path = './script.sh'  # Path to your Bash script
    command = ['bash', bash_script_path, arg1, arg2]  # Call with bash explicitly
    subprocess.run(command, check=True)

# Example usage
run_bash_script("hello", "tanner is here")
