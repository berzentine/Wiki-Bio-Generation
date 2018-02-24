import torch.nn as nn
from torch.autograd import Variable

class RNNModel(nn.Module):
    def __init__(self, dropout, vocab_size, input_dim, num_layers, hidden_size):
        super(RNNModel,self).__init__()
        self.drop = nn.Dropout(dropout)
        self.rnn_type = 'LSTM'
        self.hidden_size = hidden_size
        self.nlayers = num_layers

        self.encoder = nn.Embeddings(vocab_size,input_dim) # just a simple embedding lookup is encoder
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers, bidirectional=False, batch_first=True)
        self.decoder = nn.Linear(hidden_size, vocab_size) # run a linear layer to decode the LSTM
        self.init_weights() # just called once to randomly intialize params
        self.init_hidden() # init_hidden is called at start of every epoch again ?

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange,initrange)
        self.encoder.bias.data.fill_(0) # why not for this in the code?

        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange,initrange)

    def forward(self,x, hidden_size):
        encoded = self.drop(self.encoder(x)) # when to use dropping?
        output, hidden = self.rnn(encoded, hidden_size)
        output = self.drop(output) # when to use dropping?
        # output dimensions are vocab_size X embedding X batch ?
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2))) # what view are we changing
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden # what is this?

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data # what is this?
        return Variable(weight.new(self.nlayers, batch_size, self.hidden_size).zero_())
