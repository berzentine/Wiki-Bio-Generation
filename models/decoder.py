import torch
import torch.nn as nn
from models.dual_attention import DualAttention


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, encoder_hidden_size, num_layers, vocab_size, field_embed_size):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=False, batch_first=True)
        self.attn_layer = DualAttention(encoder_hidden_size, hidden_size, field_embed_size)
        self.lin1 = nn.Linear(2*hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.lin2 = nn.Linear(hidden_size, vocab_size)
        pass

    def forward(self, input, hidden, encoder_hidden, input_z):
        output, hidden = self.lstm(input, hidden)
        attn_vectors = self.attn_layer(output, encoder_hidden, input_z)
        concat_v = torch.cat((output, attn_vectors), 2)
        out = self.tanh(self.lin1(concat_v))
        out = self.lin2(out)
        return out, hidden