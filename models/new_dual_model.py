import torch
import torch.nn as nn
from models.decoder import Decoder
from models.encoder import Encoder
from torch.autograd import Variable

from models.lstm_dual_attention_decoder import LSTMDualAttention


class Seq2SeqDualModel(nn.Module):
    def __init__(self, sent_vocab_size, field_vocab_size, ppos_vocab_size, pneg_vocab_size, value_vocab_size, sent_embed_size, field_embed_size, \
                 value_embed_size, ppos_embed_size, pneg_embed_size, encoder_hidden_size, decoder_hidden_size, decoder_num_layer, verbose, cuda_var, x):
        super(Seq2SeqDualModel, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.sent_lookup = nn.Embedding(sent_vocab_size, sent_embed_size)
        self.field_lookup = nn.Embedding(field_vocab_size, field_embed_size)
        self.ppos_lookup = nn.Embedding(ppos_vocab_size, ppos_embed_size)
        self.pneg_lookup = nn.Embedding(pneg_vocab_size, pneg_embed_size)
        self.field_rep_embed_size = field_embed_size+ppos_embed_size+pneg_embed_size
        #self.decoder = nn.LSTM(input_size=sent_embed_size, hidden_size=encoder_hidden_size, num_layers=1, bidirectional=False, batch_first=True)
        self.encoder = nn.LSTM(input_size=sent_embed_size+self.field_rep_embed_size, hidden_size=decoder_hidden_size//2, num_layers=1, bidirectional=True, batch_first=True)
        self.decoder = LSTMDualAttention(input_size=sent_embed_size, field_rep_size=self.field_rep_embed_size, hidden_size=decoder_hidden_size, encoder_hidden_size=encoder_hidden_size, batch_first=True)
        self.linear_out = nn.Linear(encoder_hidden_size, sent_vocab_size)
        self.verbose = verbose
        self.cuda_var = cuda_var
        self.init_weights()
        self.x = nn.Parameter(torch.zeros(1), requires_grad=True)

    def init_weights(self):
        torch.nn.init.xavier_uniform(self.sent_lookup.weight)
        torch.nn.init.xavier_uniform(self.field_lookup.weight)
        torch.nn.init.xavier_uniform(self.ppos_lookup.weight)
        torch.nn.init.xavier_uniform(self.pneg_lookup.weight)

    # def forward(self, sent, value, field, ppos, pneg, batch_size, value_mask):
    #     input_d = self.sent_lookup(value)
    #     input_z = torch.cat((self.field_lookup(field), self.ppos_lookup(ppos), self.pneg_lookup(pneg)), 2)
    #     input = torch.cat((input_d,input_z), 2)
    #     encoder_output, encoder_hidden = self.encoder(input, None)
    #     #encoder_hidden = None
    #     sent = self.sent_lookup(sent)
    #     decoder_output, decoder_hidden = self.decoder(sent, encoder_hidden)
    #     decoder_output = self.linear_out(decoder_output)
    #     return decoder_output, decoder_hidden


    def forward_with_attn(self, sent, value, field, ppos, pneg, batch_size, value_mask, align_prob, epsilon=0.001):

        input_d = self.sent_lookup(value)
        input_z = torch.cat((self.field_lookup(field), self.ppos_lookup(ppos), self.pneg_lookup(pneg)), 2)
        input = torch.cat((input_d,input_z), 2)
        encoder_output, encoder_hidden = self.encoder(input, None)
        #encoder_hidden = None
        sent = self.sent_lookup(sent)
        encoder_hidden = (encoder_hidden[0].view(1, encoder_hidden[0].size(1), encoder_hidden[0].size(0)*encoder_hidden[0].size(2)), encoder_hidden[1].view(1, encoder_hidden[1].size(1), encoder_hidden[1].size(0)*encoder_hidden[1].size(2)))
        encoder_hidden = (encoder_hidden[0].squeeze(0),encoder_hidden[1].squeeze(0))
        decoder_output, decoder_hidden, attn = self.decoder.forward(sent, encoder_hidden, input_z, encoder_output)
        decoder_output = self.linear_out(decoder_output)
        logsoftmax = nn.LogSoftmax(dim=2)
        decoder_output = logsoftmax(decoder_output)
        #print(len(attn), decoder_output.size(), attn[0].size()) # (78, (32L, 78L, 20003L), (32L, 100L))

        # stack the attention vector in the second dimension -> basically convert the list of 78 attn vectors to (32, 78,100 ) single matrix
        attn = torch.stack(attn, dim=1) # (32L, 78L, 100L)
        m = nn.Sigmoid()
        lamda = m(self.x)
        #print('lambda', lamda)
        #for param in self.parameters():
        #    print param.size()
        #epsilon  = 0.001
        #dim(align_prob) = batch X vocab X table length
        #align_prob = Variable(torch.rand(attn.size(0), decoder_output.size(2), attn.size(2)))   # (32L, 20003L, 100L)
        if self.cuda_var:
            align_prob = align_prob.cuda()
        p_lex = torch.bmm(attn, align_prob) # do attn . align_prob' -> (32L, 78L, 20003L) same dimensions as decoder output
        p_mod = decoder_output
        p_bias = lamda*p_lex + (1-lamda)*p_mod + epsilon # (32L, 78L, 20003L)
        return p_bias, decoder_hidden # should return the changed and weighted decoder output and not this output
        # should return decoder_output + LfAi + e
