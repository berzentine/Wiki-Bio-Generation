import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LSTMUnit(nn.Module):
    def __init__(self, hidden_size, embed_size, verbose):
        super(LSTMUnit, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        # Weights W_*x
        self.lin1 = nn.Linear(embed_size+hidden_size, 4 * hidden_size)
        self.verbose = verbose

    def init_hidden(self, batch_size, hidden_dim):
        # return Variable(torch.zeros(batch_size, 2*hidden_dim))
        return (Variable(torch.zeros(batch_size, hidden_dim)), Variable(torch.zeros(batch_size, hidden_dim)))


    def recurrence(self, x_t, h_t_1, c_t_1):
        inp = torch.cat((x_t, h_t_1), dim=1)
        gates_vanilla = self.lin1(inp)
        ingate, forgetgate, cellgate, outgate = gates_vanilla.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate+1.0)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        c_t = (forgetgate * c_t_1) + (ingate * cellgate)
        h_t = outgate * F.tanh(c_t)

        return h_t, c_t


    def forward(self, input, hidden): # input vector, h_0 intialized as 0's and same for cell state
        output = []
        steps = range(input.size(1))  # input_d = batch X seq_length X dim
        hidden, cell_state = hidden
        for i in steps:
            hidden, cell_state = self.recurrence(input[:,i,:], hidden, cell_state)
            output.append(hidden)  # output[t][1] = hidden = batch x hidden ;; same for cell_state
        #output = torch.cat(output, 0).view(input.size(0), *output[0].size())
        return output, (hidden,cell_state)
