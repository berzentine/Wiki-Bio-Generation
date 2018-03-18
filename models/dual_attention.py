import torch
import torch.nn as nn
import torch.nn.functional as F


class DualAttention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, field_embed_size):
        self.lin1 = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        self.lin2 = nn.Linear(decoder_hidden_size, decoder_hidden_size)
        self.lin3 = nn.Linear(field_embed_size, decoder_hidden_size)
        self.lin4 = nn.Linear(decoder_hidden_size, decoder_hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, output, encoder_hidden, input_z):
        out_hs = self.tanh(self.lin1(encoder_hidden))
        out_fds = self.tanh(self.lin3(input_z))
        a = []
        out2 = self.tanh(self.lin2(output))
        g1 = torch.mul(out_hs, out2)
        alpha_t = F.softmax(g1, 0)
        out2 = self.tanh(self.lin4(output))
        g2 = torch.mul(out_fds, out2)
        beta_t = F.softmax(g2, 0)
        q = alpha_t*beta_t
        qn = torch.norm(q, p=1, dim=0).detach()
        gamma = q.div(qn.expand_as(q))
        attn_vector = torch.sum(gamma*encoder_hidden)
        return attn_vector