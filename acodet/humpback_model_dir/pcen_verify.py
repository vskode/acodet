import torch
import numpy as np
import tensorflow as tf

from .leaf_pcen import PCEN as tfPCEN
from .torch_PCEN import PCEN as torchPCEN

# --- 1. The PyTorch Implementation (From previous response) ---
# [Insert the PCEN class and pcen_ema function from above here]

# --- 2. Comparison Logic ---
def verify_pcen_parity():
    # Hyperparameters
    batch_size, time_steps, channels = 2, 100, 64
    alpha, smooth_coef, delta, root = 0.8, 0.05, 2.0, 2.0
    
    # Create identical input data
    np_input = np.random.uniform(0.1, 1.0, (batch_size, time_steps, channels)).astype(np.float32)
    torch_input = torch.from_numpy(np_input)
    tf_input = tf.convert_to_tensor(np_input)

    # --- Initialize TensorFlow Model ---
    tf_layer = tfPCEN(alpha=alpha, smooth_coef=smooth_coef, delta=delta, root=root, trainable=True)
    # Build to initialize weights
    _ = tf_layer(tf_input) 
    
    # --- Initialize PyTorch Model ---
    pt_layer = torchPCEN(num_channels=channels, alpha=alpha, smooth_coef=smooth_coef, 
                    delta=delta, root=root, trainable=True)

    # --- Ensure Weights match exactly ---
    # (In case the initializers vary slightly in precision)
    with torch.no_grad():
        pt_layer.alpha.copy_(torch.from_numpy(tf_layer.alpha.numpy()))
        pt_layer.delta.copy_(torch.from_numpy(tf_layer.delta.numpy()))
        pt_layer.root.copy_(torch.from_numpy(tf_layer.root.numpy()))

    # --- Execution ---
    tf_output = tf_layer(tf_input).numpy()
    
    pt_layer.eval()
    with torch.no_grad():
        pt_output = pt_layer(torch_input).numpy()

    # --- Comparison ---
    max_diff = np.max(np.abs(tf_output - pt_output))
    is_close = np.allclose(tf_output, pt_output, atol=1e-5)

    print(f"Verification Results:")
    print(f"- Max Absolute Difference: {max_diff:.2e}")
    print(f"- Outputs match within tolerance: {is_close}")

    if not is_close:
        # Debugging the EMA specifically if things go wrong
        print("\nChecking EMA parity...")
        # (Compare the intermediate ema_smoother results here if needed)
