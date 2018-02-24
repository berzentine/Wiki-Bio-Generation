import torch.nn as nn
from torch.autograd import Variable
#from BiLSTMEncoder import BiLSTMEncoder
from models.BiLSTMEncoder import BiLSTMEncoder as BiLSTMEncoder

class ConditionedLM(nn.Module):
    def __init__(self, dropout, vocab_size, embed_size, num_layers, hidden_size):
        super(ConditionedLM,self).__init__()
        #self.drop = nn.Dropout(dropout)
        self.rnn_type = 'LSTM'
        self.hidden_size = hidden_size
        self.nlayers = num_layers
        self.table_encoder = BiLSTMEncoder(self.hidden_size//2, num_layers=num_layers, vocab_size=vocab_size, embed_size=embed_size)
        self.encoder = nn.Embedding(vocab_size,embed_size) # just a simple embedding lookup is encoder
        self.rnn = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=False, batch_first=True)
        self.decoder = nn.Linear(hidden_size, vocab_size) # run a linear layer to decode the LSTM
        self.init_weights() # just called once to randomly intialize params
        #self.init_hidden() # init_hidden is called at start of every epoch again ?

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange,initrange)
        # self.encoder.bias.data.fill_(0) # why not for this in the code?

        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange,initrange)

    def forward(self, x, table, initial_lm_hidden, initial_encoder_hidden):
        #encoded = self.drop(self.encoder(x)) # when to use dropping?
        encoded = self.encoder(x) # when to use dropping?
        table_encoded, table_hidden = self.table_encoder(table, initial_encoder_hidden)
        output, hidden = self.rnn(encoded, initial_lm_hidden)
        #output = self.drop(output) # when to use dropping?
        output = output.contiguous()
        # output dimensions are vocab_size X embedding X batch ?
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2))) # what view are we changing
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden # what is this?

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, batch_size, self.hidden_size).zero_()),
                Variable(weight.new(self.nlayers, batch_size, self.hidden_size).zero_()))
        #return autograd.Variable(weight.new(num_layers, batch_size, hidden_dim).zero_())
