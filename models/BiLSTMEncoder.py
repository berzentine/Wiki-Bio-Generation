import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchwordemb
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMEncoder(nn.Module):
    def __init__(self, dropout,  vocab_size, embed_size, num_layers, encoder_hidden_size):
        super(BiLSTMEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_dim = encoder_hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)

        self.biLSTM = nn.LSTM(input_size=self.embed_size, hidden_size=encoder_hidden_size//2, num_layers=num_layers, bidirectional=True, batch_first=True)
        #self.initialize_weights()

    def forward(self, input, hidden):
        emdout = self.embed(input)
        #print(emdout.size())
        #for i in range(len(hidden)):
        #    hidden[i] = hidden[i].permute(1, 0, 2).contiguous() 
        out, hidden = self.biLSTM(emdout, hidden)
        print("BiLSTM:")
        print(hidden[0].size(), hidden[1].size())
        #out = out.contiguous()
        #hidden = list(hidden)
        #for i in range(len(hidden)):
        #    hidden[i] = hidden[i].permute(1, 0, 2).contiguous()  
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return [autograd.Variable(weight.new(2*self.num_layers, batch_size, self.hidden_dim//2).zero_()),
                autograd.Variable(weight.new(2*self.num_layers, batch_size, self.hidden_dim//2).zero_())]
        #return autograd.Variable(weight.new(num_layers, batch_size, hidden_dim).zero_())
