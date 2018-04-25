import torch
import torch.nn as nn
import torch.nn.functional as F


class DualAttention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, field_rep_size):
        super(DualAttention, self).__init__()
        self.lin_encoder = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        self.lin_decoder_alpha = nn.Linear(decoder_hidden_size, decoder_hidden_size)
        self.lin_decoder_beta = nn.Linear(decoder_hidden_size, decoder_hidden_size)
        self.lin_z = nn.Linear(field_rep_size, decoder_hidden_size)

        self.linear_out = nn.Linear(decoder_hidden_size * 2, decoder_hidden_size, bias=False)

    def forward(self, input, context, input_z):
        projected_context = F.tanh(self.lin_encoder(context)) #batch * encoder_seqL * encoder_hidden
        projected_input = F.tanh(self.lin_decoder_alpha(input)).unsqueeze(2)  # batch * decoder_hidden * 1
        attn_dot = torch.bmm(projected_context, projected_input).squeeze(2) #batch * encoder_seqL
        alpha = F.softmax(attn_dot, dim=1)
        # reshaped_attn = attn.view(attn.size(0), 1, attn.size(1))  #batch * 1 * encoder_seqL

        projected_inputz = F.tanh(self.lin_z(input_z)) #batch * encoder_seqL * encoder_hidden
        projected_input = F.tanh(self.lin_decoder_beta(input)).unsqueeze(2)  # batch * decoder_hidden * 1
        attn_dot = torch.bmm(projected_inputz, projected_input).squeeze(2) #batch * encoder_seqL
        beta = F.softmax(attn_dot, dim=1)

        gamma = alpha*beta #batch * encoder_seqL
        l1_norm = torch.sum(gamma, dim=1, keepdim=True)
        gamma = gamma.div(l1_norm) #batch * encoder_seqL

        reshaped_attn = gamma.view(gamma.size(0), 1, gamma.size(1))  #batch * 1 * encoder_seqL

        weighted_context = torch.bmm(reshaped_attn, context).squeeze(1) #batch * encoder_hidden
        h_tilde = torch.cat((weighted_context, input), 1)
        h_tilde = F.tanh(self.linear_out(h_tilde))

        return h_tilde, gamma
        
    def forward_with_context(self, input, context, input_z):
        projected_context = F.tanh(self.lin_encoder(context)) #batch * encoder_seqL * encoder_hidden
        projected_input = F.tanh(self.lin_decoder_alpha(input)).unsqueeze(2)  # batch * decoder_hidden * 1
        attn_dot = torch.bmm(projected_context, projected_input).squeeze(2) #batch * encoder_seqL
        alpha = F.softmax(attn_dot, dim=1)
        # reshaped_attn = attn.view(attn.size(0), 1, attn.size(1))  #batch * 1 * encoder_seqL

        projected_inputz = F.tanh(self.lin_z(input_z)) #batch * encoder_seqL * encoder_hidden
        projected_input = F.tanh(self.lin_decoder_beta(input)).unsqueeze(2)  # batch * decoder_hidden * 1
        attn_dot = torch.bmm(projected_inputz, projected_input).squeeze(2) #batch * encoder_seqL
        beta = F.softmax(attn_dot, dim=1)

        gamma = alpha*beta #batch * encoder_seqL
        l1_norm = torch.sum(gamma, dim=1, keepdim=True)
        gamma = gamma.div(l1_norm) #batch * encoder_seqL

        reshaped_attn = gamma.view(gamma.size(0), 1, gamma.size(1))  #batch * 1 * encoder_seqL

        weighted_context = torch.bmm(reshaped_attn, context).squeeze(1) #batch * encoder_hidden
        h_tilde = torch.cat((weighted_context, input), 1)
        h_tilde = F.tanh(self.linear_out(h_tilde))

        return h_tilde, gamma, weighted_context
