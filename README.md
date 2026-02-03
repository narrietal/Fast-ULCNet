# Fast-ULCNet

Official repository of **Fast-ULCNet**.

The paper is available [here](https://arxiv.org/abs/2601.14925).

A demo with online examples is available [here](https://narrietal.github.io/Fast-ULCNet/).

This repository contains the code to build the Comfi-FastGRNN and Fast-ULCNet model in Tensorflow 2+ and Pytorch.  
A `requirements.txt` file is provided for setting up the environment.

Additionally, the Comfi-FastGRNN layer is available as a pip package, making it easy to integrate into any TensorFlow or PyTorch model.

---

## Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

## Build model

### Tensorflow
```bash
python network/tensorflow_version/FastULCNet.py
```
### Pytorch
```bash
python network/pytorch_version/FastULCNetTorch.py
```
### Unit test
A simple unit test code is provided to compare the Comfi-FastGRNN implementations between Tensorflow and Pytorch.
```bash
python unit_tests/unit_test_tensorflow_torch.py
```

## Comfi-FastGRNN as a pip package
To simplify integration, the Comfi-FastGRNN layer is also distributed as a pip package.
### Tensorflow
```bash
pip install comfi-fast-grnn-tensorflow 
```
```python
import tensorflow as tf
from comfi_fast_grnn_tensorflow import ComfiFastGRNN 

comfi_fgrnn = ComfiFastGRNN(
    units=64, 
    return_sequences=False
),
```

### Pytorch
```bash
pip install comfi-fast-grnn-torch
```
```python
import torch
from comfi_fast_grnn_torch import ComfiFastGRNN 

comfi_fgrnn = ComfiFastGRNN(
    input_size=32, 
    hidden_size=64, 
    num_layers=1
)
```
## To do list
- [x] Fast-ULCNet Pytorch implementation
- [x] Python package of Comfi-FastGRNN for both Tensorflow and Pytorch

## Citation
If you use Fast-ULCNet to inspire your research, please cite the paper: (TBD)

