# Fast-ULCNet

Official repository of **Fast-ULCNet**.

The paper is available [here](https://arxiv.org/abs/2601.14925).

A demo with online examples is available [here](https://narrietal.github.io/Fast-ULCNet/).

This repository contains the code to build the Comfi-FastGRNN and Fast-ULCNet model in Tensorflow 2+ and Pytorch.  
A `requirements.txt` file is provided for setting up the environment.  

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

## To do list
- [x] Fast-ULCNet Pytorch implementation
- [ ] Python package of Comfi-FastGRNN for both Tensorflow and Pytorch

## Citation
If you use Fast-ULCNet to inspire your research, please cite the paper: (TBD)


