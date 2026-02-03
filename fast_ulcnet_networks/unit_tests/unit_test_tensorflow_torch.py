import numpy as np
import torch
import tensorflow as tf
from tensorflow.keras.layers import RNN, Bidirectional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from comfi_fast_grnn_tensorflow import ComfiFastGRNN as ComfiFastGRNNTF
from comfi_fast_grnn_torch import ComfiFastGRNN  as ComfiFastGRNNTorch
torch.manual_seed(42)

# -------------------------------
# Weight Copy Function
# -------------------------------
def copy_weights_tf_to_torch(tf_cell, torch_cell):
    """Copy weights from one direction cell."""
    with torch.no_grad():
        if hasattr(tf_cell, "w_matrix"):
            torch_cell.w_matrix.copy_(torch.tensor(tf_cell.w_matrix.numpy()))
        else:
            torch_cell.w_matrix_1.copy_(torch.tensor(tf_cell.w_matrix_1.numpy()))
            torch_cell.w_matrix_2.copy_(torch.tensor(tf_cell.w_matrix_2.numpy()))

        if hasattr(tf_cell, "u_matrix"):
            torch_cell.u_matrix.copy_(torch.tensor(tf_cell.u_matrix.numpy()))
        else:
            torch_cell.u_matrix_1.copy_(torch.tensor(tf_cell.u_matrix_1.numpy()))
            torch_cell.u_matrix_2.copy_(torch.tensor(tf_cell.u_matrix_2.numpy()))

        torch_cell.bias_gate.copy_(torch.tensor(tf_cell.bias_gate.numpy()))
        torch_cell.bias_update.copy_(torch.tensor(tf_cell.bias_update_gate.numpy()))
        torch_cell.zeta.copy_(torch.tensor(tf_cell.zeta.numpy()))
        torch_cell.nu.copy_(torch.tensor(tf_cell.nu.numpy()))
        torch_cell.lambd.copy_(torch.tensor(tf_cell.lambd.numpy()))
        torch_cell.gamma.copy_(torch.tensor(tf_cell.gamma.numpy()))

# -------------------------------
# Random input tensor parameters
# -------------------------------
hidden_size = 4
input_size = 3
seq_len = 5
batch_size = 2

# -------------------------------
# TensorFlow Setup and run inference
# -------------------------------
# Create input for TF
x_tf = tf.random.normal((batch_size, seq_len, input_size))
# Build TF RNN
tf_rnn = Bidirectional(ComfiFastGRNNTF(units=hidden_size, return_sequences=True))
# Run inference
tf_rnn_output = tf_rnn(x_tf)

# -------------------------------
# PyTorch Setup
# -------------------------------
# Create input for Torch
x_torch = torch.tensor(x_tf.numpy(), dtype=torch.float32)
# Build Torch RNN
torch_rnn = ComfiFastGRNNTorch(
    input_size=input_size,
    hidden_size=hidden_size,
    bidirectional=True,
)
# Copy weights from TF to Torch.
fw_cell_tf = tf_rnn.forward_layer.cell
bw_cell_tf = tf_rnn.backward_layer.cell
fw_cell_torch = torch_rnn.cells_fwd[0]
bw_cell_torch = torch_rnn.cells_bwd[0]
copy_weights_tf_to_torch(fw_cell_tf, fw_cell_torch)
copy_weights_tf_to_torch(bw_cell_tf, bw_cell_torch)
# Run inference
torch_rnn_output, _ = torch_rnn(x_torch)

# -------------------------------
# Compare results
# -------------------------------
output_tf = tf_rnn_output.numpy()
output_torch = torch_rnn_output.detach().numpy()

print("TensorFlow Output:")
print(output_tf)
print("\nPyTorch Output:")
print(output_torch)
print("\nL2 Difference:", np.linalg.norm(output_tf - output_torch))