import torch
import torch.nn as nn
from torch.autograd import Variable
#from BiLSTMEncoder import BiLSTMEncoder
from models.BiLSTMEncoder import BiLSTMEncoder as BiLSTMEncoder

class ConditionedLM(nn.Module):
    def __init__(self, dropout, vocab_size, embed_size, num_layers, lm_hidden_size, encoder_hidden_size):
        super(ConditionedLM,self).__init__()
        #self.drop = nn.Dropout(dropout)
        self.rnn_type = 'LSTM'
        self.hidden_size = lm_hidden_size
        self.nlayers = num_layers
        self.table_encoder = BiLSTMEncoder(encoder_hidden_size, num_layers=num_layers, vocab_size=vocab_size, embed_size=embed_size)
        self.encoder = nn.Embedding(vocab_size,embed_size) # just a simple embedding lookup is encoder
        self.rnn = nn.LSTM(input_size=embed_size, hidden_size=lm_hidden_size, num_layers=num_layers, bidirectional=False, batch_first=True)
        self.decoder = nn.Linear(lm_hidden_size, vocab_size) # run a linear layer to decode the LSTM
        self.init_weights() # just called once to randomly intialize params

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange,initrange)
        # self.encoder.bias.data.fill_(0) # why not for this in the code?

        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange,initrange)

    def forward(self, x, table, initial_lm_hidden, initial_encoder_hidden):
        encoded = self.encoder(x)
        table_encoded, table_hidden = self.table_encoder(table, initial_encoder_hidden)
        hidden = (table_hidden[0].view(-1, table_hidden[0].size(0)*table_hidden[0].size(2)).unsqueeze(0), table_hidden[1].view(-1, table_hidden[1].size(0)*table_hidden[1].size(2)).unsqueeze(0) )
        new_hidden = []
        new_out = []
        for i in range(encoded.size(1)):
            d = encoded[:, i, :].contiguous()
            out, hidden = self.rnn(d.view(encoded.size(0),1,encoded.size(2)), hidden)
            out=out.contiguous()
            new_out.append(torch.squeeze(out))
        output = torch.stack(new_out, dim=0)
        output = output.contiguous()
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1))

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, batch_size, self.hidden_size).zero_()),
                Variable(weight.new(self.nlayers, batch_size, self.hidden_size).zero_()))
