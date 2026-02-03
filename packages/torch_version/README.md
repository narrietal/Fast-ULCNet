# comfi-fast-grnn-torch
Implements Comfi-FastGRNN in torch.

## Usage

The `ComfiFastGRNN` module is designed to be a drop-in replacement for standard PyTorch RNN layers (like `nn.LSTM` or `nn.GRU`), but with added support for low-rank factorization and complementary filtering.

### Basic Implementation
Here is how to use the layer with default settings in a standard training loop:

```python
import torch
from comfi_fast_grnn_torch import ComfiFastGRNN 

comfi_fgrnn = ComfiFastGRNN(
    input_size=32, 
    hidden_size=64, 
    num_layers=1
)
```
