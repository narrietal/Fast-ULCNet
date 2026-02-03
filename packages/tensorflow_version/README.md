# comfi-fast-grnn-tensorflow
Implements Comfi-FastGRNN in tensorflow.

## Usage

You can use the `ComfiFastGRNN` layer just like any standard Keras RNN layer (e.g., `LSTM`, `GRU`). It supports the Sequential and Functional APIs.

### Basic Implementation
The simplest way to use the layer with default settings:

```python
import tensorflow as tf
from comfi_fast_grnn_tensorflow import ComfiFastGRNN  
    
# Comfi-FastGRNN layer
comfi_fgrnn = ComfiFastGRNN(
    units=64, 
    return_sequences=False
)
```
