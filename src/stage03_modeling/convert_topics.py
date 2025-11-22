import numpy as np
import json
import os

def convert_npy_to_json():
    print("[CONVERT] ========== Starting convert_npy_to_json() ==========")
    # Get the current working directory (where the script is invoked)
    print("[CONVERT] Getting current working directory...")
    cwd = os.getcwd()
    print(f"[CONVERT] Current working directory: {cwd}")

    # Get the current directory name
    print("[CONVERT] Getting current directory name...")
    current_dir = os.path.basename(cwd)
    print(f"[CONVERT] Current directory name: {current_dir}")

    # Get the grandparent directory name
    print("[CONVERT] Getting grandparent directory name...")
    grandparent_dir = os.path.basename(os.path.dirname(os.path.dirname(cwd)))
    print(f"[CONVERT] Grandparent directory name: {grandparent_dir}")

    # Define the file path for the topics.npy in the current working directory
    print("[CONVERT] Constructing NPY file path...")
    npy_file = os.path.join(cwd, 'topics.npy')
    print(f"[CONVERT] NPY file path: {npy_file}")

    # Check if the file exists
    print("[CONVERT] Checking if NPY file exists...")
    if not os.path.exists(npy_file):
        print(f"[CONVERT] ❌ {npy_file} not found in the current directory.")
        return
    print(f"[CONVERT] ✓ NPY file exists")

    # Load the .npy file
    print("[CONVERT] Loading NPY file...")
    try:
        data = np.load(npy_file, allow_pickle=True)  # allow_pickle=True to load objects
        print(f"[CONVERT] ✓ NPY file loaded")
        print(f"[CONVERT] Data type: {type(data)}")
        if hasattr(data, 'shape'):
            print(f"[CONVERT] Data shape: {data.shape}")
        elif isinstance(data, (list, tuple)):
            print(f"[CONVERT] Data length: {len(data)}")
    except Exception as e:
        print(f"[CONVERT] ❌ Error loading {npy_file}: {e}")
        import traceback
        traceback.print_exc()
        return

    # Convert to a JSON serializable format
    print("[CONVERT] Converting to JSON serializable format...")
    try:
        json_data = json.dumps(data.tolist(), indent=4)
        print(f"[CONVERT] ✓ JSON conversion completed")
        print(f"[CONVERT] JSON data length: {len(json_data):,} characters")
    except TypeError as e:
        print(f"[CONVERT] ❌ Error converting to JSON: {e}")
        import traceback
        traceback.print_exc()
        return

    # Define the output file name as grandparent__current__topics.json
    print("[CONVERT] Constructing output filename...")
    json_filename = f"{grandparent_dir}__{current_dir}__topics.json"
    print(f"[CONVERT] Output filename: {json_filename}")

    # Save as a JSON file in the current working directory
    print("[CONVERT] Saving JSON file...")
    json_file = os.path.join(cwd, json_filename)
    print(f"[CONVERT] JSON file path: {json_file}")
    with open(json_file, 'w') as f:
        f.write(json_data)
    print(f"[CONVERT] ✓ JSON file written")

    file_size = os.path.getsize(json_file) / (1024**2)
    print(f"[CONVERT] Conversion complete. JSON saved as {json_file} ({file_size:.2f} MB).")
    print("[CONVERT] ========== convert_npy_to_json() completed ==========")

if __name__ == "__main__":
    convert_npy_to_json()

