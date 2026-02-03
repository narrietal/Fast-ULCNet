import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Auxiliar non-linear generation function
# -------------------------------
def gen_non_linearity(A, non_linearity):
    """
    Returns required activation for a tensor based on the inputs
    """
    if non_linearity == "tanh":
        return torch.tanh(A)
    elif non_linearity == "sigmoid":
        return torch.sigmoid(A)
    elif non_linearity == "relu":
        return torch.relu(A)
    elif non_linearity == "quantTanh":
        return torch.clamp(A, -1.0, 1.0)
    elif non_linearity == "quantSigm":
        A = (A + 1.0) / 2.0
        return torch.clamp(A, 0.0, 1.0)
    elif non_linearity == "quantSigm4":
        A = (A + 2.0) / 4.0
        return torch.clamp(A, 0.0, 1.0)
    elif callable(non_linearity):
        return non_linearity(A)
    else:
        raise ValueError(
            "non_linearity must be one of ['tanh', 'sigmoid', 'relu', 'quantTanh', 'quantSigm', 'quantSigm4'] or callable"
        )

# -------------------------------
# Comfi-FastGRNN cell
# -------------------------------
class ComfiFastGRNNCellTorch(nn.Module):
    '''
    Comfi-FastGRNN Cell

    This class is imported from the official FastGRNN cell code to Pytorch syntax, and
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
        input_size,
        hidden_size,
        gate_non_linearity="sigmoid",
        update_non_linearity="tanh",
        w_rank=None,
        u_rank=None,
        zeta_init=1.0,
        nu_init=-4.0,
        lambda_init=0.0,
        gamma_init=0.999,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate_non_linearity = gate_non_linearity
        self.update_non_linearity = update_non_linearity
        self.w_rank = w_rank
        self.u_rank = u_rank

        # --- Weight definitions ---
        if w_rank is None:
            self.w_matrix = nn.Parameter(torch.randn(input_size, hidden_size) * 0.1)
        else:
            self.w_matrix_1 = nn.Parameter(torch.randn(input_size, w_rank) * 0.1)
            self.w_matrix_2 = nn.Parameter(torch.randn(w_rank, hidden_size) * 0.1)

        if u_rank is None:
            self.u_matrix = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        else:
            self.u_matrix_1 = nn.Parameter(torch.randn(hidden_size, u_rank) * 0.1)
            self.u_matrix_2 = nn.Parameter(torch.randn(u_rank, hidden_size) * 0.1)

        # --- Biases ---
        self.bias_gate = nn.Parameter(torch.ones(1, hidden_size))
        self.bias_update = nn.Parameter(torch.ones(1, hidden_size))

        # --- Scalars ---
        self.zeta = nn.Parameter(torch.tensor([[zeta_init]], dtype=torch.float32))
        self.nu = nn.Parameter(torch.tensor([[nu_init]], dtype=torch.float32))
        self.lambd = nn.Parameter(torch.tensor([lambda_init], dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor([gamma_init], dtype=torch.float32))

    def forward(self, x, h_prev):
        # Compute W*x
        if self.w_rank is None:
            W = x @ self.w_matrix
        else:
            W = x @ self.w_matrix_1 @ self.w_matrix_2

        # Compute U*h_prev
        if self.u_rank is None:
            U = h_prev @ self.u_matrix
        else:
            U = h_prev @ self.u_matrix_1 @ self.u_matrix_2

        # Gates
        z = gen_non_linearity(W + U + self.bias_gate, self.gate_non_linearity)
        h_hat = gen_non_linearity(W + U + self.bias_update, self.update_non_linearity)

        # FastGRNN update
        h = z * h_prev + (torch.sigmoid(self.zeta) * (1 - z) + torch.sigmoid(self.nu)) * h_hat

        # Comfi-FastGRNN update
        gamma_clamped = torch.clamp(self.gamma, 0.0, 1.0)
        h_t_comfi = gamma_clamped * h + (1 - gamma_clamped) * self.lambd

        return h_t_comfi
    
# -------------------------------
# Comfi-FastGRNN layer
# -------------------------------
class ComfiFastGRNNTorch(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        # ---- Cell arguments (explicitly exposed) ----
        gate_non_linearity: str = "sigmoid",
        update_non_linearity: str = "tanh",
        w_rank: int | None = None,
        u_rank: int | None = None,
        zeta_init: float = 1.0,
        nu_init: float = -4.0,
        lambda_init: float = 0.0,
        gamma_init: float = 0.999,
    ):
        super().__init__()

        # ---- Layer config ----
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # ---- Cell config (stored for export / repr / checkpoint clarity) ----
        self.gate_non_linearity = gate_non_linearity
        self.update_non_linearity = update_non_linearity
        self.w_rank = w_rank
        self.u_rank = u_rank
        self.zeta_init = zeta_init
        self.nu_init = nu_init
        self.lambda_init = lambda_init
        self.gamma_init = gamma_init

        # ---- Cells ----
        self.cells_fwd = nn.ModuleList()
        self.cells_bwd = nn.ModuleList() if bidirectional else None

        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size * self.num_directions

            self.cells_fwd.append(
                ComfiFastGRNNCellTorch(
                    input_size=in_size,
                    hidden_size=hidden_size,
                    gate_non_linearity=gate_non_linearity,
                    update_non_linearity=update_non_linearity,
                    w_rank=w_rank,
                    u_rank=u_rank,
                    zeta_init=zeta_init,
                    nu_init=nu_init,
                    lambda_init=lambda_init,
                    gamma_init=gamma_init,
                )
            )

            if bidirectional:
                self.cells_bwd.append(
                    ComfiFastGRNNCellTorch(
                        input_size=in_size,
                        hidden_size=hidden_size,
                        gate_non_linearity=gate_non_linearity,
                        update_non_linearity=update_non_linearity,
                        w_rank=w_rank,
                        u_rank=u_rank,
                        zeta_init=zeta_init,
                        nu_init=nu_init,
                        lambda_init=lambda_init,
                        gamma_init=gamma_init,
                    )
                )

    def forward(self, x, h0=None):
        """
        x:  (batch, seq_len, input_size) if batch_first=True
        h0: (num_layers * num_directions, batch, hidden_size)

        Returns:
          output: (batch, seq_len, hidden_size * num_directions)
          h_n:    (num_layers * num_directions, batch, hidden_size)
        """
        if not self.batch_first:
            x = x.transpose(0, 1)

        batch_size, seq_len, _ = x.size()

        if h0 is None:
            h0 = x.new_zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size,
            )

        layer_input = x
        h_n = []

        for layer in range(self.num_layers):
            fw_cell = self.cells_fwd[layer]
            h_fw = h0[layer * self.num_directions + 0]
            fw_outs = []

            # ---- forward ----
            for t in range(seq_len):
                h_fw = fw_cell(layer_input[:, t, :], h_fw)
                fw_outs.append(h_fw.unsqueeze(1))

            fw_out = torch.cat(fw_outs, dim=1)

            if self.bidirectional:
                bw_cell = self.cells_bwd[layer]
                h_bw = h0[layer * self.num_directions + 1]
                bw_outs = []

                # ---- backward ----
                for t in reversed(range(seq_len)):
                    h_bw = bw_cell(layer_input[:, t, :], h_bw)
                    bw_outs.append(h_bw.unsqueeze(1))

                bw_outs.reverse()
                bw_out = torch.cat(bw_outs, dim=1)

                layer_out = torch.cat([fw_out, bw_out], dim=2)
                h_n.extend([h_fw, h_bw])
            else:
                layer_out = fw_out
                h_n.append(h_fw)

            if self.dropout > 0.0 and layer < self.num_layers - 1:
                layer_out = F.dropout(layer_out, p=self.dropout, training=self.training)

            layer_input = layer_out

        output = layer_input
        h_n = torch.stack(h_n, dim=0)

        if not self.batch_first:
            output = output.transpose(0, 1)

        return output, h_n
