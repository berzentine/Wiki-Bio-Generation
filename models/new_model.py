import torch
import torch.nn as nn
from models.decoder import Decoder
from models.dot_attention import LSTMAttentionDot
from models.encoder import Encoder
from torch.autograd import Variable


class Seq2SeqModelNew(nn.Module):
    def __init__(self, sent_vocab_size, field_vocab_size, ppos_vocab_size, pneg_vocab_size, value_vocab_size, sent_embed_size, field_embed_size, \
                 value_embed_size, ppos_embed_size, pneg_embed_size, encoder_hidden_size, decoder_hidden_size, decoder_num_layer, verbose, cuda_var):
        super(Seq2SeqModelNew, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.sent_lookup = nn.Embedding(sent_vocab_size, sent_embed_size)
        self.field_lookup = nn.Embedding(field_vocab_size, field_embed_size)
        self.ppos_lookup = nn.Embedding(ppos_vocab_size, ppos_embed_size)
        self.pneg_lookup = nn.Embedding(pneg_vocab_size, pneg_embed_size)
        self.field_rep_embed_size = field_embed_size+ppos_embed_size+pneg_embed_size
        self.decoder = nn.LSTM(input_size=sent_embed_size, hidden_size=encoder_hidden_size, num_layers=1, bidirectional=False, batch_first=True)
        self.encoder = nn.LSTM(input_size=sent_embed_size+self.field_rep_embed_size, hidden_size=decoder_hidden_size, num_layers=1, bidirectional=False, batch_first=True)
        self.decoder = LSTMAttentionDot(input_size=sent_embed_size, hidden_size=decoder_hidden_size, batch_first=True)
        self.linear_out = nn.Linear(encoder_hidden_size, sent_vocab_size)
        self.verbose = verbose
        self.cuda_var = cuda_var
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform(self.sent_lookup.weight)
        torch.nn.init.xavier_uniform(self.field_lookup.weight)
        torch.nn.init.xavier_uniform(self.ppos_lookup.weight)
        torch.nn.init.xavier_uniform(self.pneg_lookup.weight)

    def forward(self, sent, value, field, ppos, pneg, batch_size, value_mask):
        input_d = self.sent_lookup(value)
        input_z = torch.cat((self.field_lookup(field), self.ppos_lookup(ppos), self.pneg_lookup(pneg)), 2)
        input = torch.cat((input_d,input_z), 2)
        encoder_output, encoder_hidden = self.encoder(input, None)
        #encoder_hidden = None
        sent = self.sent_lookup(sent)
        decoder_output, decoder_hidden = self.decoder(sent, encoder_hidden)
        decoder_output = self.linear_out(decoder_output)
        return decoder_output, decoder_hidden


    def forward_with_attn(self, sent, value, field, ppos, pneg, batch_size, value_mask):
        input_d = self.sent_lookup(value)
        input_z = torch.cat((self.field_lookup(field), self.ppos_lookup(ppos), self.pneg_lookup(pneg)), 2)
        input = torch.cat((input_d,input_z), 2)
        encoder_output, encoder_hidden = self.encoder(input, None)
        #encoder_hidden = None
        sent = self.sent_lookup(sent)
        encoder_hidden = (encoder_hidden[0].squeeze(0),encoder_hidden[1].squeeze(0))
        decoder_output, decoder_hidden, attn = self.decoder.forward(sent, encoder_hidden, encoder_output)
        decoder_output = self.linear_out(decoder_output)
        return decoder_output, decoder_hidden
