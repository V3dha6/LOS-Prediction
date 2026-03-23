import os

# Where is the script looking?
current_dir = os.getcwd()
print(f"Current Working Directory (where Python is running from): {current_dir}")

# List all files in the current directory
print("Files in this directory:")
print(os.listdir(current_dir))

# Check if files exist in the parent directory
parent_dir = os.path.dirname(current_dir)
print(f"\nFiles in parent directory ({parent_dir}):")
try:
    print(os.listdir(parent_dir))
except FileNotFoundError:
    print("Parent directory not found.")