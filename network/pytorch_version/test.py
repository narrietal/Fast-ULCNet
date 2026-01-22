import torch
import numpy as np
import yaml
import os

# Assuming you saved the classes above in their respective files
from FastULCNetTorch import FastULCNetTorch
# from FastULCNet import FastULCNet  # Import your TF version here

def create_mock_config():
    config = {
        'data_parameters': {
            'fs': 16000,
            'block_len': 512,
            'block_shift': 128,
            'hann_window': True,
            'compression_factor': 0.5
        },
        'model_parameters': {
            'bidirectional_frnn_units': 64,
            'sub_band_rnn_units': 128,
            'CRM_type': 'masking'
        },
        'training_parameters': {}
    }
    return config

def test_architectures():
    config = create_mock_config()
    batch_size = 2
    samples = 16000 # 1 second of audio
    
    print("--- Testing PyTorch Model ---")
    pt_model = FastULCNetTorch(config)
    dummy_input_pt = torch.randn(batch_size, samples)
    
    with torch.no_grad():
        pt_output = pt_model(dummy_input_pt)
    
    print(f"PyTorch Input Shape: {dummy_input_pt.shape}")
    print(f"PyTorch Output Shape: {pt_output.shape}")
    
    # Verify expected frequency bins (N/2 + 1)
    expected_freq = config['data_parameters']['block_len'] // 2 + 1
    assert pt_output.shape[-1] == expected_freq, "Frequency dimension mismatch!"

    print("\n--- Structural Verification ---")
    # You can compare the number of parameters
    total_params = sum(p.numel() for p in pt_model.parameters())
    print(f"Total PyTorch Parameters: {total_params:,}")

    print("\nSuccess: Architecture conversion verified for shape consistency.")

if __name__ == '__main__':
    test_architectures()