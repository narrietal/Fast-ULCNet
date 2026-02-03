# fast-ulcnet-tensorflow
Implements Comfi-FastGRNN in tensorflow.

## Usage

You can use the `ComfiFastGRNN` layer just like any standard Keras RNN layer (e.g., `LSTM`, `GRU`). It supports the Sequential and Functional APIs.

### Basic Implementation
The simplest way to use the layer with default settings:

```python
import tensorflow as tf
from comfi_fast_grnn_tensorflow import ComfiFastGRNN  

# Define a Sequential model
model = tf.keras.Sequential([
    # Input shape: (Timesteps, Features)
    tf.keras.layers.Input(shape=(100, 32)), 
    
    # Comfi-FastGRNN layer
    ComfiFastGRNN(
        units=64, 
        return_sequences=False
    ),
    
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()
```
