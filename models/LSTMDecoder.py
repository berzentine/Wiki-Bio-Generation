import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTMDecoder(nn.Module):
    def __init__(self, dropout, vocab_size, embed_size, num_layers, decoder_hidden_size):
        super(LSTMDecoder,self).__init__()
        self.rnn_type = 'LSTM'
        self.hidden_size = decoder_hidden_size
        self.nlayers = num_layers
        self.encoder = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(input_size=embed_size, hidden_size=decoder_hidden_size, num_layers=num_layers, bidirectional=False, batch_first=True)
        self.decoder = nn.Linear(decoder_hidden_size, vocab_size)
        #self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange,initrange)
        # self.encoder.bias.data.fill_(0) # why not for this in the code?

        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange,initrange)

    def forward(self, input, hidden):
        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded, hidden)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden


    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, batch_size, self.hidden_size).zero_()),
                Variable(weight.new(self.nlayers, batch_size, self.hidden_size).zero_()))
        #return autograd.Variable(weight.new(num_layers, batch_size, hidden_dim).zero_())
