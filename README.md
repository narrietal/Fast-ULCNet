# Fast-ULCNet

Official repository of **Fast-ULCNet**.

The paper is available [here](https://arxiv.org/abs/2601.14925).

A demo with online examples is available [here](https://narrietal.github.io/Fast-ULCNet/).

This repository contains the code to build the Comfi-FastGRNN and Fast-ULCNet model in Tensorflow 2+ and Pytorch.  
A `requirements.txt` file is provided for setting up the environment.  

---

## Manual Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

## Pip Installation
To install FastULCNet and Comfi-FastGRNN you can run:
```bash
pip install fast-ulcnet-torch
pip install fast-ulcnet-tensorflow
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

## Citation
If you use Fast-ULCNet to inspire your research, please cite the paper: (TBD)
