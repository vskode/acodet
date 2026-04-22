import tensorflow as tf
import numpy as np
import re

def inject_weights(model, npz_path, verbose=False):
    # 1. Load the Ground Truth
    if verbose:
        print(f"Loading NPZ from {npz_path}...")
    data = np.load(npz_path)
    
    # Get the list of keys actually present in the file
    # (We convert to a set for O(1) lookups)
    gt_keys = set(data.files)

    if verbose:
        print("Starting Exact Match Injection...")
    assigned_count = 0
    
    # 2. Iterate over the New Model's variables
    for variable in model.variables:
        new_name = variable.path
        
        # --- THE CLEANING LOGIC ---
        # This converts your New Model name into the Ground Truth format
        # based strictly on the lists you provided.
        
        # A. Remove auto-generated IDs from main_path/residual_path
        # Pattern: matches "main_path_32" -> "main_path"
        clean_name = re.sub(r'(main_path|residual_path)_\d+', r'\1', new_name)
        
        # B. Remove auto-generated IDs from simple_rnn
        # Pattern: matches "simple_rnn_2" -> "simple_rnn"
        clean_name = re.sub(r'simple_rnn_\d+', 'simple_rnn', clean_name)
        
        # C. Ensure it ends with :0 (Ground Truth has it, Model doesn't)
        if not clean_name.endswith(':0'):
            clean_name = clean_name + ':0'
            
        # --- END LOGIC ---

        # 3. Perform the Injection
        if clean_name in gt_keys:
            gt_value = data[clean_name]
            
            # shape check
            if variable.shape == gt_value.shape:
                variable.assign(gt_value)
                assigned_count += 1
            else:
                print(f"[SHAPE FAIL] Name matched but shapes differ for {new_name}")
                print(f"  Model: {variable.shape} vs NPZ: {gt_value.shape}")
        else:
            if 'seed' in clean_name:
                continue
            # If we are here, something is wrong with the NPZ or the logic
            print(f"[MISSING] Parsed key not found in NPZ: {clean_name}")
            print(f"  Original Model Var: {new_name}")

    if verbose:
        print("-" * 30)
        print(f"Injection Complete. Assigned {assigned_count}/{len(model.variables)} variables.")

# --- USAGE ---
# model = ... (your compiled model)
# inject_weights_hardcoded(model, "ground_truth.npz")