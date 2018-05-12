import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, z_size, hidden_size, batch_size, embed_size, bias=True):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.bias = bias
        self.z_size = z_size
        # Weights W_*h
        self.weight_ih = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_oh = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_fh = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_ctildeh = Parameter(torch.Tensor(hidden_size, hidden_size))
        # Weights W_*x
        self.weight_ix = Parameter(torch.Tensor(embed_size, hidden_size))
        self.weight_ox = Parameter(torch.Tensor(embed_size, hidden_size))
        self.weight_fx = Parameter(torch.Tensor(embed_size, hidden_size))
        self.weight_ctildex = Parameter(torch.Tensor(embed_size, hidden_size))
        self.weight_lx = Parameter(torch.Tensor(z_size, hidden_size))
        self.weight_zhatx = Parameter(torch.Tensor(z_size, hidden_size))
        # Bias
        if bias:
            self.bias_i = Parameter(torch.Tensor(hidden_size))
            self.bias_f = Parameter(torch.Tensor(hidden_size))
            self.bias_o = Parameter(torch.Tensor(hidden_size))
            self.bias_l = Parameter(torch.Tensor(hidden_size))
            self.bias_zhat = Parameter(torch.Tensor(hidden_size))
            self.bias_ctilde = Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias_i', None)
            self.register_parameter('bias_f', None)
            self.register_parameter('bias_o', None)
            self.register_parameter('bias_l', None)
            self.register_parameter('bias_zhat', None)
            self.register_parameter('bias_ctilde', None)

    def forward(self, input, hidden, cell_state): # input vector, h_0 intialized as 0's and same for cell state
        def recurrence(d_t, z_t, h_t_1, c_t_1):
            i_t = F.sigmoid(F.linear(h_t_1, self.weight_ih, None) + F.linear(d_t, self.weight_ix, None) + self.bias_i)
            return d_t_1, z_t_1, h_t, c_t
