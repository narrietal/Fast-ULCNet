# Fast-ULCNet

Official repository of **Fast-ULCNet**.

The paper is available [here](https://arxiv.org/abs/2601.14925).

A demo with online examples is available [here](https://narrietal.github.io/Fast-ULCNet/).

This repository contains the code to build the Comfi-FastGRNN and Fast-ULCNet model in Tensorflow 2+ and Pytorch.  

The Comfi-FastGRNN layer is available as a pip package, making it easy to integrate into any TensorFlow or PyTorch model.

---

## Installation
Clone the repository, create an environment and install the dependencies.

### Install requirements
```bash
pip install -r requirements.txt
```

### Install Comfi-FastGRNN Tensorflow layer
From pypi:
```bash
pip install comfi-fast-grnn-tensorflow 
```

### Install Comfi-FastGRNN Pytorch layer
From pypi:
```bash
pip install comfi-fast-grnn-torch
```

## Build model

### Tensorflow
```bash
python fast_ulcnet_networks/tensorflow_version/FastULCNet.py
```
### Pytorch
```bash
python fast_ulcnet_networks/pytorch_version/FastULCNetTorch.py
```
### Unit test
A simple unit test code is provided to compare the Comfi-FastGRNN implementations between Tensorflow and Pytorch.
```bash
python fast_ulcnet_networks/unit_tests/unit_test_tensorflow_torch.py
```

## To do list
- [x] Fast-ULCNet Pytorch implementation
- [x] Python package of Comfi-FastGRNN for both Tensorflow and Pytorch

## Citation
If you use Fast-ULCNet to inspire your research, please cite the paper: (TBD)

