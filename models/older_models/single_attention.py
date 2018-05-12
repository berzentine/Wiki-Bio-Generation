import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleAttention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super(SingleAttention, self).__init__()
        self.lin_encoder = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        self.lin_decoder = nn.Linear(decoder_hidden_size, decoder_hidden_size)
        self.linear_out = nn.Linear(decoder_hidden_size * 2, decoder_hidden_size, bias=False)

    def forward(self, input, context):
        projected_context = F.tanh(self.lin_encoder(context)) #batch * encoder_seqL * encoder_hidden
        projected_input = F.tanh(self.lin_decoder(input)).unsqueeze(2)  # batch * decoder_hidden * 1
        #print(projected_context.size(), input.size(
        attn_dot = torch.bmm(projected_context, projected_input).squeeze(2) #batch * encoder_seqL
        attn = F.softmax(attn_dot, dim=1)
        reshaped_attn = attn.view(attn.size(0), 1, attn.size(1))  #batch * 1 * encoder_seqL
        
        weighted_context = torch.bmm(reshaped_attn, context).squeeze(1) #batch * encoder_hidden

        h_tilde = torch.cat((weighted_context, input), 1)
        h_tilde = F.tanh(self.linear_out(h_tilde))

        return h_tilde, attn
