import tensorflow as tf
from tensorflow.keras.layers import AbstractRNNCell
from tensorflow.keras import backend
from tensorflow.keras.initializers import RandomNormal, Constant, Ones
from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.keras.layers import RNN


def gen_non_linearity(A, non_linearity):
    '''
    Returns required activation for a tensor based on the inputs
    non_linearity is either a callable or a value in
    ['tanh', 'sigmoid', 'relu', 'quantTanh', 'quantSigm', 'quantSigm4']
    '''
    
    if non_linearity == "tanh":
        return backend.tanh(A)
    elif non_linearity == "sigmoid":
        return backend.sigmoid(A)
    elif non_linearity == "relu":
        return backend.maximum(A, 0.0)
    elif non_linearity == "quantTanh":
        return backend.maximum(backend.minimum(A, 1.0), -1.0)
    elif non_linearity == "quantSigm":
        A = (A + 1.0) / 2.0
        return backend.maximum(backend.minimum(A, 1.0), 0.0)
    elif non_linearity == "quantSigm4":
        A = (A + 2.0) / 4.0
        return backend.maximum(backend.minimum(A, 1.0), 0.0)
    else:
        # non_linearity is a user specified function
        if not callable(non_linearity):
            raise ValueError("non_linearity is either a callable or a value: ['tanh', 'sigmoid', 'relu', 'quantTanh', 'quantSigm', 'quantSigm4']")
        return non_linearity(A)

# -------------------------------
# Comfi-FastGRNN cell
# -------------------------------
class ComfiFastGRNNCell(AbstractRNNCell):

    '''
    Comfi-FastGRNN Cell

    This class is upgraded from the official FastGRNN cell code to Tensorflow 2.0 syntax, and
    it is extended with the trainable complementary filter aproach suggested in our paper.

    Original FastGRNN cell code available in: https://github.com/microsoft/EdgeML/tree/master    
    
    The cell has both Full Rank and Low Rank Formulations and
    multiple activation functions for the gates.

    hidden_size = # hidden units

    gate_non_linearity = nonlinearity for the gate can be chosen from
    [tanh, sigmoid, relu, quantTanh, quantSigm]

    update_non_linearity = nonlinearity for final rnn update
    can be chosen from [tanh, sigmoid, relu, quantTanh, quantSigm]

    w_rank = rank of W matrix (creates two matrices if not None)
    u_rank = rank of U matrix (creates two matrices if not None)
    zeta_init = init for zeta, the scale param
    nu_init = init for nu, the translation param
    lambda_init = init value for lambda, the CF drift modulation parameter
    gamma_init = init value for gamma, the CF hidden state contribution parameter

    Equations of the RNN state update:

    z_t = gate_nl(W + Uh_{t-1} + B_g)
    h_t^ = update_nl(W + Uh_{t-1} + B_h)
    h_t = z_t*h_{t-1} + (sigmoid(zeta)(1-z_t) + sigmoid(nu))*h_t^
    h_t_comfi = gamma*h_t + (1-gamma)*lambda

    W and U can further parameterised into low rank version by
    W = matmul(W_1, W_2) and U = matmul(U_1, U_2)
    '''

    def __init__(
        self,
        hidden_size,
        gate_non_linearity="sigmoid",
        update_non_linearity="tanh",
        w_rank=None,
        u_rank=None,
        zeta_init=1.0,
        nu_init=-4.0,
        lambda_init=0.0,
        gamma_init=0.999,
        name="FastGRNN",
        **kwargs):

        super(ComfiFastGRNNCell, self).__init__(**kwargs)
        self._hidden_size = hidden_size
        self._gate_non_linearity = gate_non_linearity
        self._update_non_linearity = update_non_linearity
        self._num_weight_matrices = [1, 1]
        self._w_rank = w_rank
        self._u_rank = u_rank
        self._zeta_init = zeta_init
        self._nu_init = nu_init
        self._lambda_init = lambda_init
        self._gamma_init = gamma_init
        
        if w_rank is not None:
            self._num_weight_matrices[0] += 1
        if u_rank is not None:
            self._num_weight_matrices[1] += 1
        self._name = name

    @property
    def state_size(self):
        return self._hidden_size

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def gate_non_linearity(self):
        return self._gate_non_linearity

    @property
    def update_non_linearity(self):
        return self._update_non_linearity

    @property
    def w_rank(self):
        return self._w_rank

    @property
    def u_rank(self):
        return self._u_rank

    @property
    def num_weight_matrices(self):
        return self._num_weight_matrices

    @property
    def name(self):
        return self._name

    @property
    def cellType(self):
        return "Comfi-FastGRNN"

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if self._w_rank is None:
            w_init = RandomNormal(mean=0.0, stddev=0.1)
            self.w_matrix = self.add_weight(
            shape=(input_dim, self._hidden_size),
            name="w_matrix",
            initializer=w_init
        )
        else:
            w_init_1 = RandomNormal(mean=0.0, stddev=0.1)
            w_init_2 = RandomNormal(mean=0.0, stddev=0.1)
            self.w_matrix_1 = self.add_weight(
            shape=(input_dim, self._w_rank),
            name="w_matrix_1",
            initializer=w_init_1
            )
            self.w_matrix_2 = self.add_weight(
            shape=(self._w_rank, self._hidden_size),
            name="w_matrix_2",
            initializer=w_init_2
            )

        if self._u_rank is None:
            u_init = RandomNormal(mean=0.0, stddev=0.1)
            self.u_matrix = self.add_weight(
            shape=(self._hidden_size, self._hidden_size),
            name="u_matrix",
            initializer=u_init
            )
        else:
            u_init_1 = RandomNormal(mean=0.0, stddev=0.1)
            u_init_2 = RandomNormal(mean=0.0, stddev=0.1)
            self.u_matrix_1 = self.add_weight(
            shape=(self._hidden_size, self._u_rank),
            name="u_matrix_1",
            initializer=u_init_1
            )
            self.u_matrix_2 = self.add_weight(
            shape=(self._u_rank, self._hidden_size),
            name="u_matrix_2",
            initializer=u_init_2
            )

        zeta_init = Constant(self._zeta_init)
        nu_init = Constant(self._nu_init)
        lambda_init = Constant(self._lambda_init)
        gamma_init = Constant(self._gamma_init)
        bias_gate_initializer = Ones()
        bias_update_initializer = Ones()
        
        self.zeta = self.add_weight(
        shape=(1,1),
        name="zeta",
        initializer=zeta_init
        )
        self.nu = self.add_weight(
        shape=(1,1),
        name="nu",
        initializer=nu_init
        )
        
        self.bias_gate = self.add_weight(
        shape=(1, self._hidden_size),
        name="bias_gate",
        initializer=bias_gate_initializer
        )

        self.bias_update_gate = self.add_weight(
        shape=(1, self._hidden_size),
        name="bias_update",
        initializer=bias_update_initializer
        )
        
        self.lambd = self.add_weight(
        shape=(1,),
        name="lambd",
        initializer=lambda_init,
        trainable=True,
        )
        
        self.gamma= self.add_weight(
        shape=(1,),
        name="gamma",
        trainable=True,
        initializer=gamma_init,
        constraint=MinMaxNorm(min_value=0.0, max_value=1.0) # Note: We can either limit its value range here or use a sigmoid function later on.
        )

        self.built = True

    def call(self, inputs, states):
        h_tm1 = states[0]
        if self._w_rank is None:
            W = backend.dot(inputs, self.w_matrix)
        else:
            w_tmp = backend.dot(inputs, self.w_matrix_1)
            W = backend.dot(w_tmp, self.w_matrix_2)

        if self._u_rank is None:
            U = backend.dot(h_tm1, self.u_matrix)
        else:
            u_tmp = backend.dot(h_tm1, self.u_matrix_1)
            U = backend.dot(u_tmp, self.u_matrix_2)

        z = gen_non_linearity( (W + U + self.bias_gate), self._gate_non_linearity )
        
        h_hat = gen_non_linearity( (W + U + self.bias_update_gate), self._update_non_linearity )
        
        h = ((z * h_tm1) + (backend.sigmoid(self.zeta) * (1.0 - z) + backend.sigmoid(self.nu)) * h_hat)

        # Comfi-FastGRNN state update
        h_t_comfi = h*self.gamma + (1-self.gamma)*self.lambd

        return h_t_comfi, h_t_comfi

    def get_config(self):
        config = {
        "hidden_size": self._hidden_size,
        "gate_non_linearity": self._gate_non_linearity,
        "update_non_linearity":self._update_non_linearity,
        "w_rank": self._w_rank,
        "u_rank":self._u_rank,
        "zeta_init":self._zeta_init,
        "nu_init":self._nu_init,
        "lambda_init":self._lambda_init,
        "gamma_init":self._gamma_init,
        "name":self._name
        }

        base_config = super(ComfiFastGRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# -------------------------------
# Comfi-FastGRNN layer
# -------------------------------
class ComfiFastGRNN(tf.keras.layers.RNN):

    def __init__(
        self,
        units,
        gate_non_linearity="sigmoid",
        update_non_linearity="tanh",
        w_rank=None,
        u_rank=None,
        zeta_init=1.0,
        nu_init=-4.0,
        lambda_init=0.0,
        gamma_init=0.999,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        unroll=False,
        **kwargs,
    ):

        self.hidden_size = units
        self.gate_non_linearity = gate_non_linearity
        self.update_non_linearity = update_non_linearity
        self.w_rank = w_rank
        self.u_rank = u_rank
        self.zeta_init = zeta_init
        self.nu_init = nu_init
        self.lambda_init = lambda_init
        self.gamma_init = gamma_init

        cell = ComfiFastGRNNCell(
            hidden_size=units,
            gate_non_linearity=gate_non_linearity,
            update_non_linearity=update_non_linearity,
            w_rank=w_rank,
            u_rank=u_rank,
            zeta_init=zeta_init,
            nu_init=nu_init,
            lambda_init=lambda_init,
            gamma_init=gamma_init,
        )

        super().__init__(
            cell=cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs,
        )

    def get_config(self):
        config = super().get_config()

        # Remove serialized cell
        config.pop("cell", None)

        config.update({
            "units": self.hidden_size,
            "gate_non_linearity": self.gate_non_linearity,
            "update_non_linearity": self.update_non_linearity,
            "w_rank": self.w_rank,
            "u_rank": self.u_rank,
            "zeta_init": self.zeta_init,
            "nu_init": self.nu_init,
            "lambda_init": self.lambda_init,
            "gamma_init": self.gamma_init,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
