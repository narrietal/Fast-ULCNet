
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Lambda, Input, Activation, RNN, Bidirectional, Concatenate, SeparableConv2D, MaxPool2D, Conv2D, Layer
import math
import libsegmenter
from CRM_tensorflow import ComplexRatioMask
from ComfiFastGRNN import ComfiFastGRNNCell
import yaml

class STFTLayer(Layer):
    """
    Custom TensorFlow Layer for computing STFT.
    """
    def __init__(self, block_len, block_shift, window_fn=None, **kwargs):
        super(STFTLayer, self).__init__(**kwargs)
        self.block_len = block_len
        self.block_shift = block_shift
        self.window_fn = window_fn

    def call(self, inputs):
        frames = tf.signal.frame(inputs, self.block_len, self.block_shift)
        if self.window_fn is not None: frames *= self.window_fn
        stft = tf.signal.rfft(frames)
        return stft

    def get_config(self):
        config = super(STFTLayer, self).get_config()
        config.update({
            "block_len": self.block_len,
            "block_shift": self.block_shift,
            "window_fn": self.window_fn,
        })
        return config

class ChannelWiseFeatureReorientation(tf.keras.layers.Layer):
    """
    Custom TensorFlow Layer for computing Channel-wise Feature Reorientation 
    technique described in https://arxiv.org/html/2312.08132v1.
    """
    def __init__(self,  input_freq_dim=257, **kwargs):
        super(ChannelWiseFeatureReorientation, self).__init__(**kwargs)
        self.input_freq_dim = int(input_freq_dim)
        self.window_size = 48
        overlap = 0.33
        self.hop_size = math.ceil(self.window_size  * (1-overlap))
        self.n_bands = math.ceil(((input_freq_dim - self.window_size ) / self.hop_size) + 1)

    def call(self, inputs):
        subbands = []
        for i in range(self.n_bands):
            start = int(i * self.hop_size)
            end = int(start + self.window_size)       
            if end > self.input_freq_dim:
                subband = inputs[:, :, start:self.input_freq_dim]
                n_zeros = end - self.input_freq_dim
                zeros_to_append = tf.zeros((tf.shape(inputs)[0], tf.shape(inputs)[1], n_zeros), dtype=inputs.dtype)
                subband = tf.concat([subband, zeros_to_append], axis=-1) 
            else:
                subband = inputs[:, :, start:end] 
            subbands.append(subband)
        subband_spectrogram = tf.stack(subbands, axis=2)
        return subband_spectrogram

class FastULCNet():
    """
    Fast-ULCNet network class.
    """
    def __init__(self, config) -> None:
        # defining data parameters
        self.config = config
        self.data_parameters = config['data_parameters']
        self.fs = self.data_parameters['fs']
        self.blockLen = self.data_parameters['block_len']
        self.block_shift = self.data_parameters['block_shift']
        self.hann_window = self.data_parameters['hann_window']
        self.compression_factor = self.data_parameters['compression_factor']
        # defining model parameters
        self.model_parameters = config['model_parameters']
        self.bidirectional_frnn_units = self.model_parameters['bidirectional_frnn_units']
        self.sub_band_rnn_units = self.model_parameters['sub_band_rnn_units']
        #Initialize Complex Ratio Mask layer
        crm_type = self.model_parameters['CRM_type']
        self.crm_layer = ComplexRatioMask(masking_mode=crm_type)
        # ChannelWiseFeatureReorientation
        self.chWiseFeatureReorientation = ChannelWiseFeatureReorientation(input_freq_dim=(self.blockLen/2)+1) 
        if self.hann_window: self.window  = libsegmenter.WindowSelector("hann75", "wola", self.blockLen).analysis_window
        # defining training parameters
        self.training_parameters = config['training_parameters']
        # Create stft layer
        self.stft_layer = STFTLayer(self.blockLen, self.block_shift, self.window)

    def feature_preprocessing(self, x):
        '''
        Method to extract power law compressed noisy magnitude spectrogram and phase features based on the UCLNet paper.
        This layer takes a complex spectrogram and returns magnitude and phase values.
        '''
        real_part = tf.math.real(x)
        imag_part = tf.math.imag(x) 
        compressed_real_part = tf.math.sign(real_part)*tf.math.pow(tf.math.abs(real_part), self.compression_factor)
        compressed_imag_part = tf.math.sign(imag_part)*tf.math.pow(tf.math.abs(imag_part), self.compression_factor)
        magnitude = tf.math.sqrt(tf.math.square(compressed_real_part) + tf.math.square(compressed_imag_part))
        phase = tf.math.atan2(compressed_imag_part,compressed_real_part)
        return magnitude, phase, compressed_real_part, compressed_imag_part
    
    def intermediate_feature_computation(self, intermediate_mask, input_phase):
        '''        
        This layer takes a real-valued mask and the phase feature computed from the original stft and computes the intermediate features described in the UCLNet paper.
        Lastly, it returns the features stacked along the channel dimension.
        '''
        real_inter_feature = intermediate_mask * tf.math.cos(input_phase)
        imag_inter_feature = intermediate_mask * tf.math.sin(input_phase)
        inter_feature = tf.stack([real_inter_feature,imag_inter_feature], axis=3)      
        return inter_feature
    
    def power_law_decompression(self, x):
        '''
        Method to compensate for the power law compression applied to the input features as described in the UCLNet paper.
        The layer takes a complex speech estimation spectrogram and decompresses its real and imaginary parts.
        ''' 
        real_part = tf.math.real(x)
        imag_part = tf.math.imag(x)
        decompressed_real_part = tf.math.sign(real_part)*tf.math.pow(tf.math.abs(real_part), 1/self.compression_factor)
        decompressed_imag_part = tf.math.sign(imag_part)*tf.math.pow(tf.math.abs(imag_part), 1/self.compression_factor)
        decompressed_speech = tf.complex(decompressed_real_part,decompressed_imag_part)
        return decompressed_speech
    
    def conv_block(self, x):
        '''
        Conv block as described on the paper.
        It works as a bottleneck -> Number of filters increases while the frequency dimension decreases after the max pool operations.
        '''
        x = SeparableConv2D(filters = 32, kernel_size = (1,3), padding='same', use_bias=False)(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(filters = 64, kernel_size = (1,3), padding='same', use_bias=False)(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size=(1, 2))(x)
        x = SeparableConv2D(filters = 96, kernel_size = (1,3), padding='same', use_bias=False)(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size=(1, 2))(x)
        x = SeparableConv2D(filters = 128, kernel_size = (1,3), padding='same', use_bias=False)(x)
        x = Activation('relu', name="output_relu_conv_block")(x)
        x = MaxPool2D(pool_size=(1, 2))(x)
        return x
    
    def freq_comfi_fastgrnn(self, x):
        '''
        Frequency-axis bidirectional Comfi-FastGRNN layer 
        '''
        reshaped_x = tf.reshape(x, shape=(tf.shape(x)[0]*tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]))
        output_frnn = Bidirectional(RNN(ComfiFastGRNNCell(hidden_size=self.bidirectional_frnn_units), return_sequences=True))(reshaped_x)
        reshaped_output_frnn = tf.reshape(output_frnn, (tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 2*self.bidirectional_frnn_units), name="reshape_output_frnn")
        return reshaped_output_frnn
    
    def sub_band_comfi_fastgrnn(self, x):
        '''
        Subband temporal Comfi-FastGRNN layer.
        '''
        out_rnn_1 = RNN(ComfiFastGRNNCell(hidden_size=self.sub_band_rnn_units), return_sequences=True)(x)
        out_rnn_2 = RNN(ComfiFastGRNNCell(hidden_size=self.sub_band_rnn_units), return_sequences=True)(out_rnn_1)
        return out_rnn_2
    
    def cnn_block(self, x):
        '''
        CNN block as described in the ULCNet paper
        '''
        x = Conv2D(filters=32, kernel_size=(1, 3), padding='same', use_bias=False)(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=32, kernel_size=(1, 3), padding='same', use_bias=False)(x)
        x = Activation('relu')(x)
        return x
        
    def build_model(self):
        '''
        Method to build and compile the FastULCNet network. The model takes time domain 
        batches of size (batch_size, len_in_samples) and returns a spectrogram of the estimated speech.
        '''
        ########################
        ##### FIRST STAGE ######
        ########################
        # Input layer for time signal
        input_data = Input(batch_shape=(None,None))
        # Calculate STFT
        stft_data = self.stft_layer(input_data)
        # Extract magnitude, phase, real and imaginary features
        mag, phase, real, imag = Lambda(self.feature_preprocessing)(stft_data)
        # Channel-wise ft reorientation
        learnt_representation = self.chWiseFeatureReorientation(mag)
        # Reshape input tensor to follow the channel-last order of TF: (batch_size, time_dim, channels, frequency_dim) -> (batch_size, time_dim, frequency_dim, channels)
        learnt_representation = tf.transpose(learnt_representation, perm=[0, 1, 3, 2])
        # Encode features -> Bottleneck: filters increase, freq. dimension dicreases
        output_features = self.conv_block(learnt_representation)
        # Bidirectional Freq-only Comfi-FastGRNN
        frnn_output = self.freq_comfi_fastgrnn(output_features)
        # Reduce filter size
        pointwise_conv_output = tf.keras.layers.Conv2D(64, kernel_size=(1,1), use_bias=False)(frnn_output)
        pointwise_conv_output = Activation('relu')(pointwise_conv_output)
        # Merge frequency and channel to obtain new feature dimension
        reshaped_x = tf.reshape(pointwise_conv_output, shape=(tf.shape(pointwise_conv_output)[0], tf.shape(pointwise_conv_output)[1], tf.shape(pointwise_conv_output)[2] * tf.shape(pointwise_conv_output)[3])) # reshape: (batch_dim, time_dim, freq_dim*features)
        # Split tensor into bands
        sub_band1, sub_band2 = tf.split(reshaped_x, num_or_size_splits=2, axis=2) # Split channels
        # Pass band tensors through temporal Comfi-FastGRNN blocks
        rnn_output1 = self.sub_band_comfi_fastgrnn(sub_band1)
        rnn_output2 = self.sub_band_comfi_fastgrnn(sub_band2)
        # Concatenate outputs
        concatenated_output = Concatenate(axis=-1)([rnn_output1, rnn_output2])
        # Pass through couple of FC layers
        fc_output = Dense((self.blockLen/2)+1 , activation='relu')(concatenated_output)
        mask = Dense((self.blockLen/2)+1 , activation='relu')(fc_output)
        # Compute intermediate features
        inter_feature = self.intermediate_feature_computation(mask,phase)

        ########################
        ##### SECOND STAGE #####
        ########################
        # Pass throug lightweight CNN layers
        cnn_output = self.cnn_block(inter_feature)
        # Use a pointwise convolution to have the right shape
        complex_mask = Conv2D(filters=2, kernel_size=(1, 1), use_bias=False)(cnn_output)
        complex_mask = Activation('relu')(complex_mask)
        # Reshape mask into real and imaginary components
        mask_real, mask_imag = tf.split(complex_mask, num_or_size_splits=2, axis=3)
        mask_real = tf.squeeze(mask_real, axis=-1)
        mask_imag = tf.squeeze(mask_imag, axis=-1)
        # Apply Complex Ratio Masking
        compressed_estimated_speech = self.crm_layer(real, imag, mask_real, mask_imag)
        # Decompress speech
        estimated_speech = self.power_law_decompression(compressed_estimated_speech)
        # Define model
        self.model = Model(inputs=input_data, outputs=estimated_speech)
        # show the model summary
        self.model.summary()
        
        return self.model
    
if __name__ == '__main__':
    # Load config
    with open('../config.yml', 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    
    model_runner = FastULCNet(config)

    model = model_runner.build_model()