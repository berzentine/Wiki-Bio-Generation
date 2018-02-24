import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchwordemb
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMEncoder(nn.Module):
    def __init__(self, hidden_size, num_layers, vocab_size, embed_size):
        super(BiLSTMEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)

        self.rnn = nn.LSTM(input_size=self.embed_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)

        self.initialize_weights()

    def forward(self, input, hidden, lengths):
        emdout = self.embed(input)
        out, hidden = self.biLSTM(emdout, hidden)
        #out = out.contiguous()
        return out, hidden

    def init_hidden(self, num_layers, batch_size, hidden_dim):
        weight = next(self.parameters()).data
        return (autograd.Variable(weight.new(num_layers, batch_size, 2*hidden_dim).zero_()),
                                   autograd.Variable(weight.new(num_layers, batch_size, 2*hidden_dim).zero_()))
        #return autograd.Variable(weight.new(num_layers, batch_size, hidden_dim).zero_())
