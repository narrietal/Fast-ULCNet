import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import libsegmenter
from ComfiFastGRNNTorch import ComfiFastGRNNTorch


class ComplexRatioMask(nn.Module):
    """
    PyTorch implementation of Complex Ratio Masking.
    S = (Mr * Yr - Mi * Yi) + j(Mr * Yi + Mi * Yr)
    """
    def __init__(self, masking_mode='masking'):
        super().__init__()
        self.masking_mode = masking_mode

    def forward(self, real, imag, mask_real, mask_imag):
        # real/imag: [B, T, F]
        # mask_real/mask_imag: [B, T, F]
        est_real = real * mask_real - imag * mask_imag
        est_imag = real * mask_imag + imag * mask_real
        return torch.complex(est_real, est_imag)

class STFTLayer(nn.Module):
    def __init__(self, block_len, block_shift, window=None):
        super().__init__()
        self.block_len = block_len
        self.block_shift = block_shift
        self.window = window

    def forward(self, x):
        # x: [Batch, Samples]
        stft = torch.stft(
            x, 
            n_fft=self.block_len, 
            hop_length=self.block_shift, 
            win_length=self.block_len, 
            window=self.window, 
            center=True, 
            return_complex=True
        )
        # transpose to [B, T, F]
        return stft.transpose(1, 2)

class ChannelWiseFeatureReorientation(nn.Module):
    def __init__(self, input_freq_dim=257):
        super().__init__()
        self.input_freq_dim = int(input_freq_dim)
        self.window_size = 48
        overlap = 0.33
        self.hop_size = math.ceil(self.window_size * (1 - overlap))
        self.n_bands = math.ceil(((self.input_freq_dim - self.window_size) / self.hop_size) + 1)

    def forward(self, x):
        # x: [B, T, F]
        batch_size, time_dim, freq_dim = x.shape
        subbands = []
        for i in range(self.n_bands):
            start = i * self.hop_size
            end = start + self.window_size
            if end > self.input_freq_dim:
                # Padding logic
                subband = x[:, :, start:self.input_freq_dim]
                padding = torch.zeros((batch_size, time_dim, end - self.input_freq_dim), 
                                     device=x.device, dtype=x.dtype)
                subband = torch.cat([subband, padding], dim=-1)
            else:
                subband = x[:, :, start:end]
            subbands.append(subband)
            
        print("x.shape vs output", x.shape, torch.stack(subbands, dim=2).shape)
        
        # Stack into [B, T, n_bands, window_size]
        return torch.stack(subbands, dim=2)

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1), groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(x)
    

class ConvBlock(nn.Module):
    """
    Conv block as described in the paper.
    Bottleneck structure: channels increase while frequency dimension
    decreases after max-pooling.
    """

    def __init__(self, in_channels):
        super().__init__()

        self.sepconv1 = SeparableConv2d(
            in_channels, 32
        )
        self.sepconv2 = SeparableConv2d(
            32, 64
        )
        self.sepconv3 = SeparableConv2d(
            64, 96
        )
        self.sepconv4 = SeparableConv2d(
            96, 128
        )

        self.pool = nn.MaxPool2d(kernel_size=(1, 2))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.sepconv1(x))
        x = self.relu(self.sepconv2(x))
        x = self.pool(x)

        x = self.relu(self.sepconv3(x))
        x = self.pool(x)

        x = self.relu(self.sepconv4(x))
        x = self.pool(x)

        return x

class FastULCNetTorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Data Params
        dp = config['data_parameters']
        self.block_len = dp['block_len']
        self.block_shift = dp['block_shift']
        self.compression_factor = dp['compression_factor']
        freq_dim = int(self.block_len // 2 + 1)
        mp = config['model_parameters']

        # Layers
        self.stft_layer = STFTLayer(self.block_len, self.block_shift)
        self.reorientation = ChannelWiseFeatureReorientation(input_freq_dim=freq_dim)
        
        # Conv Block
        self.conv_block = ConvBlock(in_channels=8)
        
        # RNN units        
        self.freq_rnn  = ComfiFastGRNNTorch(
                            input_size=128,
                            hidden_size=mp['bidirectional_frnn_units'],
                            bidirectional=True,
                            batch_first=True
                        )
        
        self.pointwise_conv = nn.Conv2d(2 * mp['bidirectional_frnn_units'], 64, kernel_size=(1,1), bias=False)
        
        # Sub-band RNNs
        # Pointwise output is 64 filters. Freq dim reduced by MaxPools.
        # Initial freq is n_bands. Maxpool 1x2 applied 3 times -> n_bands // 8
        reduced_freq = self.reorientation.n_bands // 8
        self.rnn_input_dim = (reduced_freq * 64) // 2
        
        #self.sub_band_rnn1 = nn.GRU(self.rnn_input_dim, mp['sub_band_rnn_units'], 
        #                            num_layers=2, batch_first=True)
        #self.sub_band_rnn2 = nn.GRU(self.rnn_input_dim, mp['sub_band_rnn_units'], 
        #                            num_layers=2, batch_first=True)
        
        self.sub_band_rnn1  = ComfiFastGRNNTorch(
                    input_size=192,
                    hidden_size=mp['sub_band_rnn_units'],
                    num_layers=2,
                    batch_first=True
                )
        
        self.sub_band_rnn2  = ComfiFastGRNNTorch(
                    input_size=192,
                    hidden_size=mp['sub_band_rnn_units'],
                    num_layers=2,
                    batch_first=True
                )
        
        # Stage 1 Outputs
        self.fc1 = nn.Linear(2 * mp['sub_band_rnn_units'], freq_dim)
        self.fc2 = nn.Linear(freq_dim, freq_dim)
        
        # Stage 2 CNN
        self.cnn_block = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.ReLU()
        )
        self.complex_mask_conv = nn.Conv2d(32, 2, kernel_size=(1, 1), bias=False)
        self.crm_layer = ComplexRatioMask(masking_mode=mp['CRM_type'])

    def feature_preprocessing(self, x):
        real, imag = x.real, x.imag
        c = self.compression_factor
        comp_real = torch.sign(real) * torch.pow(torch.abs(real), c)
        comp_imag = torch.sign(imag) * torch.pow(torch.abs(imag), c)
        mag = torch.sqrt(comp_real**2 + comp_imag**2)
        phase = torch.atan2(comp_imag, comp_real)
        return mag, phase, comp_real, comp_imag

    def forward(self, x):
        # 1. STFT and Preprocessing
        stft_data = self.stft_layer(x)
        mag, phase, real, imag = self.feature_preprocessing(stft_data)
        
        # 2. Reorientation: [B, T, F] -> [B, T, n_bands, window_size]
        features = self.reorientation(mag)
        # To [B, C, T, F] for PyTorch Conv2d
        features = features.permute(0, 2, 1, 3)
        
        print("Features after reorient:", features.shape)
        
        # 3. Conv Block with MaxPools
        x = self.conv_block(features)
        
        print("Features conv block:", x.shape)
        
        # 4. Frequency RNN
        # x shape: [B, 128, T, F_red] -> RNN expects [B*T, F_red, 128]
        B, C, T, F_red = x.shape
        x_rnn = x.permute(0, 2, 3, 1).reshape(B * T, F_red, C)
        frnn_out, _ = self.freq_rnn(x_rnn)
        print("Output frnn:", frnn_out.shape)
        frnn_out = frnn_out.view(B, T, F_red, -1).permute(0, 3, 1, 2)
        print("Output frnn PERMUTED:", frnn_out.shape)
        
        # 5. Temporal Sub-band RNNs
        x = F.relu(self.pointwise_conv(frnn_out)) # [B, 64, T, F_red]
        print("Output pointwise_conv:", x.shape)
        x = x.permute(0, 2, 3, 1).reshape(B, T, -1) # Flatten F and C
        
        print("Before subband:", x.shape)
        sub1, sub2 = torch.chunk(x, 2, dim=-1)
        print("Subband1 subband:", sub1.shape)
        r1, _ = self.sub_band_rnn1(sub1)
        r2, _ = self.sub_band_rnn2(sub2)
        concatenated = torch.cat([r1, r2], dim=-1)
        
        print("Output concatenated:", concatenated.shape)
        
        # 6. Mask Computation
        mask = F.relu(self.fc1(concatenated))
        mask = F.relu(self.fc2(mask))
        
        # 7. Intermediate Features for Stage 2
        inter_r = mask * torch.cos(phase)
        inter_i = mask * torch.sin(phase)
        inter_feat = torch.stack([inter_r, inter_i], dim=1) # [B, 2, T, F]
        
        # 8. CRM and Decompression
        cnn_out = self.cnn_block(inter_feat)
        c_mask = F.relu(self.complex_mask_conv(cnn_out))
        m_real, m_imag = c_mask[:, 0, :, :], c_mask[:, 1, :, :]
        
        est_speech_comp = self.crm_layer(real, imag, m_real, m_imag)
        
        # Decompress
        final_real, final_imag = est_speech_comp.real, est_speech_comp.imag
        inv_c = 1.0 / self.compression_factor
        dec_real = torch.sign(final_real) * torch.pow(torch.abs(final_real), inv_c)
        dec_imag = torch.sign(final_imag) * torch.pow(torch.abs(final_imag), inv_c)
        
        return torch.complex(dec_real, dec_imag)