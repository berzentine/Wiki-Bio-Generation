import torch.nn as nn
from torch.autograd import Variable

class RNNModel(nn.Module):
    def __init__(self, dropout, vocab_size, input_dim, num_layers, hidden_size):
        super(RNNModel,self).__init__()
        self.drop = nn.Dropout(dropout)
        self.rnn_type = 'LSTM'
        self.hidden_size = hidden_size
        self.nlayers = num_layers

        self.encoder = nn.Embedding(vocab_size,input_dim) # just a simple embedding lookup is encoder
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers, bidirectional=False, batch_first=True)
        self.decoder = nn.Linear(hidden_size, vocab_size) # run a linear layer to decode the LSTM
        self.init_weights() # just called once to randomly intialize params

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange,initrange)

        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange,initrange)

    def forward(self,x, hidden_size):
        encoded = self.encoder(x)
        output, hidden = self.rnn(encoded, hidden_size)
        output = output.contiguous()
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2))) # what view are we changing
        return decoded.view(output.size(0), output.size(1), decoded.size(1)) # what is this?

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, batch_size, self.hidden_size).zero_()),
                Variable(weight.new(self.nlayers, batch_size, self.hidden_size).zero_()))
