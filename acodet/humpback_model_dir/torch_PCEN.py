# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PCEN implementation, forked from google-research/leaf-audio.
This implementation was converted to torch from tensorflow using Gemini3."""

import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.jit.script
def pcen_ema(inputs: torch.Tensor, smooth_coef: float):
    """
    Implements the EMA smoothing logic: y[t] = s*x[t] + (1-s)*y[t-1]
    Matches TF SimpleRNN behavior with first-frame initialization.
    Input shape: (Batch, Time, Channels)
    """
    # Initialize state with the first frame (matches tf.gather(inputs, 0, axis=1))
    # We use a loop that is JIT-compiled for performance
    s = smooth_coef
    one_minus_s = 1.0 - s
    
    # Pre-allocate output tensor
    ema_smoother = torch.empty_like(inputs)
    state = inputs[:, 0, :]
    ema_smoother[:, 0, :] = state
    
    for t in range(1, inputs.size(1)):
        state = s * inputs[:, t, :] + one_minus_s * state
        ema_smoother[:, t, :] = state
        
    return ema_smoother

class PCEN(nn.Module):
    """
    PyTorch port of the Google Research/LEAF PCEN implementation.
    
    Expects input shape: (Batch, Time, Channels) to match your TF code.
    Note: Standard PyTorch audio often uses (Batch, Channels, Time). 
    If using the latter, permute before/after calling this.
    """
    def __init__(
        self,
        num_channels: int,
        alpha: float,
        smooth_coef: float,
        delta: float = 2.0,
        root: float = 2.0,
        floor: float = 1e-6,
        trainable: bool = False,
    ):
        super().__init__()
        self._smooth_coef = smooth_coef
        self._floor = floor
        self._trainable = trainable

        # Initialize parameters as nn.Parameter for learnability
        self.alpha = nn.Parameter(
            torch.full((num_channels,), 
            float(alpha)), 
            requires_grad=trainable
        )
        self.delta = nn.Parameter(
            torch.full((num_channels,), 
            float(delta)), 
            requires_grad=trainable
        )
        self.root = nn.Parameter(
            torch.full((num_channels,), 
            float(root)), 
            requires_grad=trainable
        )

    def forward(self, inputs: torch.Tensor):
        if len(inputs) == 3:
            inputs = inputs.unsqueeze(1)
        # 1. Constrain parameters (matches TF logic)
        alpha = torch.min(self.alpha, torch.ones_like(self.alpha))
        root = torch.max(self.root, torch.ones_like(self.root))
        
        # 2. Compute EMA Smoother
        ema_smoother = pcen_ema(inputs, self._smooth_coef)
        
        # 3. Apply PCEN formula
        # broadcasting handles the (Channels,) parameters against (Batch, Time, Channels)
        one_over_root = 1.0 / root
        
        # (inputs / (floor + ema)**alpha + delta)**(1/root) - delta**(1/root)
        output = (
            inputs / (self._floor + ema_smoother).pow(alpha) + self.delta
        ).pow(one_over_root) - self.delta.pow(one_over_root)
        
        return output.squeeze()