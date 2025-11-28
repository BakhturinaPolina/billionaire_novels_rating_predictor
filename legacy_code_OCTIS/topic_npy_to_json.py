import numpy as np
import json
import os

def convert_npy_to_json():
    # Get the current working directory (where the script is invoked)
    cwd = os.getcwd()

    # Get the current directory name
    current_dir = os.path.basename(cwd)

    # Get the grandparent directory name
    grandparent_dir = os.path.basename(os.path.dirname(os.path.dirname(cwd)))

    # Define the file path for the topics.npy in the current working directory
    npy_file = os.path.join(cwd, 'topics.npy')

    # Check if the file exists
    if not os.path.exists(npy_file):
        print(f"{npy_file} not found in the current directory.")
        return

    # Load the .npy file
    try:
        data = np.load(npy_file, allow_pickle=True)  # allow_pickle=True to load objects
    except Exception as e:
        print(f"Error loading {npy_file}: {e}")
        return

    # Convert to a JSON serializable format
    try:
        json_data = json.dumps(data.tolist(), indent=4)
    except TypeError as e:
        print(f"Error converting to JSON: {e}")
        return

    # Define the output file name as grandparent__current__topics.json
    json_filename = f"{grandparent_dir}__{current_dir}__topics.json"

    # Save as a JSON file in the current working directory
    json_file = os.path.join(cwd, json_filename)
    with open(json_file, 'w') as f:
        f.write(json_data)

    print(f"Conversion complete. JSON saved as {json_file}.")

if __name__ == "__main__":
    convert_npy_to_json()
