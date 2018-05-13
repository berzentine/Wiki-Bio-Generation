import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dual_attention import DualAttention


class LSTMDualAttention(nn.Module):
    def __init__(self, input_size, field_rep_size, hidden_size, encoder_hidden_size, batch_first=True):
        super(LSTMDualAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.batch_first = batch_first

        self.input_weights = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)

        self.attention_layer = DualAttention(encoder_hidden_size, hidden_size, field_rep_size)

    def forward(self, input, hidden, input_z, ctx, ctx_mask=None):
        def recurrence(input, hidden):
            hx, cx = hidden  # n_b x hidden_dim
            gates = self.input_weights(input) + \
                    self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            print(forgetgate.size(), cx.size())
	    hy = outgate * F.tanh(cy)  # n_b x hidden_dim
            h_tilde, alpha = self.attention_layer.forward(hy, ctx, input_z)

            return (h_tilde, cy), alpha

        if self.batch_first:
            input = input.transpose(0, 1)

        output = []
        attn = []
        steps = range(input.size(0))
        for i in steps:
            hidden, att = recurrence(input[i], hidden)
            output.append(hidden[0])
            attn.append(att)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden, attn
