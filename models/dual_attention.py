import torch
import torch.nn as nn
import torch.nn.functional as F


class DualAttention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, field_embed_size, verbose):
        super(DualAttention, self).__init__()
        self.lin1 = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        self.lin2 = nn.Linear(decoder_hidden_size, decoder_hidden_size)
        self.lin3 = nn.Linear(field_embed_size, decoder_hidden_size)
        self.lin4 = nn.Linear(decoder_hidden_size, decoder_hidden_size)
        self.tanh = nn.Tanh()
        self.verbose = verbose

    def forward(self, output, encoder_hidden, input_z):
        out_hs = self.tanh(self.lin1(encoder_hidden))
        out_hs = out_hs.view(out_hs.size(1), out_hs.size(0), out_hs.size(2))
        if self.verbose: print(out_hs.size())
        encoder_hidden = encoder_hidden.view(encoder_hidden.size(1), encoder_hidden.size(0), encoder_hidden.size(2))
        out_fds = self.tanh(self.lin3(input_z))
        if self.verbose: print(out_fds.size())
        a = []
        out2 = self.tanh(self.lin2(output))
        if self.verbose: print(out2.size())
        out3 = self.tanh(self.lin4(output))
        if self.verbose: print(out3.size())
        attn = []
        concat_vectors = []
        for i in range(out2.size(1)):
            if self.verbose: print(out2[:, i, :].unsqueeze(1).size())
            g1 = torch.mul(out_hs, out2[:, i, :].unsqueeze(1))
            if self.verbose: print(g1.size())
            alpha_t = F.softmax(g1, 1)
            if self.verbose: print(alpha_t.size())
            if self.verbose: print(out3[:, i, :].unsqueeze(1).size())
            g2 = torch.mul(out_fds, out3[:, i, :].unsqueeze(1))
            if self.verbose: print(g2.size())
            beta_t = F.softmax(g2, 1)
            if self.verbose: print(beta_t.size())
            q = alpha_t*beta_t
            if self.verbose: print(q.size())
            # qn = torch.norm(q, p=1, dim=1).detach().unsqueeze(1)
            qn = torch.sum(q, dim=1).detach().unqueeze(1)
            if self.verbose: print(qn.size())
            gamma = q.div(qn.expand_as(q))
            if self.verbose: print(gamma.size())
            attn_vector = torch.sum(gamma*encoder_hidden, dim=1)
            if self.verbose: print(attn_vector.size())
            concat_v = torch.cat((output[:, i, :], attn_vector), 1)
            attn.append(attn_vector)
            concat_vectors.append(concat_v)
        return concat_vectors, attn

    def forward_vanilla(self, output, encoder_hidden, input_z):
        out_hs = self.tanh(self.lin1(encoder_hidden))
        out_hs = out_hs.view(out_hs.size(1), out_hs.size(0), out_hs.size(2))
        if self.verbose: print(out_hs.size())
        encoder_hidden = encoder_hidden.view(encoder_hidden.size(1), encoder_hidden.size(0), encoder_hidden.size(2))
        # out_fds = self.tanh(self.lin3(input_z))
        # if self.verbose: print(out_fds.size())
        a = []
        out2 = self.tanh(self.lin2(output))
        if self.verbose: print(out2.size())
        # out3 = self.tanh(self.lin4(output))
        # if self.verbose: print(out3.size())
        attn = []
        concat_vectors = []
        for i in range(out2.size(1)):
            if self.verbose: print(out2[:, i, :].unsqueeze(1).size())
            g1 = torch.mul(out_hs, out2[:, i, :].unsqueeze(1))
            if self.verbose: print(g1.size())
            alpha_t = F.softmax(g1, 0)
            if self.verbose: print(alpha_t.size())
            # if self.verbose: print(out3[:, i, :].unsqueeze(1).size())
            # g2 = torch.mul(out_fds, out3[:, i, :].unsqueeze(1))
            # if self.verbose: print(g2.size())
            # beta_t = F.softmax(g2, 0)
            # if self.verbose: print(beta_t.size())
            # q = alpha_t*beta_t
            # if self.verbose: print(q.size())
            # qn = torch.norm(q, p=1, dim=1).detach().unsqueeze(1)
            # if self.verbose: print(qn.size())
            # gamma = q.div(qn.expand_as(q))
            # if self.verbose: print(gamma.size())
            attn_vector = torch.sum(alpha_t*encoder_hidden, dim=1)
            if self.verbose: print(attn_vector.size())
            concat_v = torch.cat((output[:, i, :], attn_vector), 1)
            attn.append(attn_vector)
            concat_vectors.append(concat_v)
        return concat_vectors, attn