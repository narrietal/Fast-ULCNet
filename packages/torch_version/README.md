# fast-ulcnet-torch
Implements FastULCNet and Comfi-FastGRNN in torch.

## Usage

The `ComfiFastGRNN` module is designed to be a drop-in replacement for standard PyTorch RNN layers (like `nn.LSTM` or `nn.GRU`), but with added support for low-rank factorization and complementary filtering.

### Basic Implementation
Here is how to use the layer with default settings in a standard training loop:

```python
import torch
from comfi_fast_grnn_torch import ComfiFastGRNN 

# 1. Initialize the layer
# batch_first=True is the default for this implementation
model = ComfiFastGRNN(
    input_size=32, 
    hidden_size=64, 
    num_layers=1
)

# 2. Create dummy input: (Batch Size, Sequence Length, Input Size)
x = torch.randn(10, 50, 32)

# 3. Forward pass
# Returns output (all timesteps) and final hidden state
output, h_n = model(x)

print(f"Output shape: {output.shape}")  # torch.Size([10, 50, 64])
print(f"Hidden state shape: {h_n.shape}") # torch.Size([1, 10, 64])
```
