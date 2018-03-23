import torch
import torch.nn as nn
from models.dual_attention import DualAttention


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, encoder_hidden_size, num_layers, vocab_size, field_embed_size, verbose):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=False, batch_first=True)
        self.attn_layer = DualAttention(encoder_hidden_size, hidden_size, field_embed_size, verbose)
        self.lin1 = nn.Linear(2*hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.lin2 = nn.Linear(hidden_size, vocab_size)
        self.verbose = verbose

    def forward(self, input, hidden, encoder_hidden, input_z):
        output, hidden = self.lstm(input, hidden)
        if self.verbose: print(output.size(), hidden[0].size(), hidden[1].size())
        concat_v, attn_vectors = self.attn_layer.forward(output, encoder_hidden, input_z)
        #concat_v = torch.cat((output, attn_vectors), 2)
        concat_v = torch.stack(concat_v, dim=0)
        out = self.tanh(self.lin1(concat_v))
        out = self.lin2(out)
        out = out.view(out.size(1), out.size(0), out.size(2))
        if self.verbose: print(out.size())
        return out, hidden

    def forward_plain(self, input, hidden, encoder_hidden, input_z):
        output, hidden = self.lstm(input, hidden)
        if self.verbose: print(output.size(), hidden[0].size(), hidden[1].size())
        # concat_v, attn_vectors = self.attn_layer.forward(output, encoder_hidden, input_z)
        # #concat_v = torch.cat((output, attn_vectors), 2)
        # concat_v = torch.stack(concat_v, dim=0)
        # out = self.tanh(self.lin1(concat_v))
        out = output
        out = self.lin2(out)
        out = out.view(out.size(1), out.size(0), out.size(2))
        if self.verbose: print(out.size())
        return out, hidden