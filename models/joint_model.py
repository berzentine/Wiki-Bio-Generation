import torch
import torch.nn as nn
from models.decoder import Decoder
from models.encoder import Encoder
from torch.autograd import Variable


class Seq2SeqModel(nn.Module):
    def __init__(self, sent_vocab_size, field_vocab_size, ppos_vocab_size, pneg_vocab_size, value_vocab_size, sent_embed_size, field_embed_size,\
                 value_embed_size, ppos_embed_size, pneg_embed_size, encoder_hidden_size, decoder_hidden_size, decoder_num_layer, verbose, cuda_var):
        super(Seq2SeqModel, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.sent_lookup = nn.Embedding(sent_vocab_size, sent_embed_size)
        self.value_lookup = nn.Embedding(value_vocab_size, value_embed_size)
        self.field_lookup = nn.Embedding(field_vocab_size, field_embed_size)
        self.ppos_lookup = nn.Embedding(ppos_vocab_size, ppos_embed_size)
        self.pneg_lookup = nn.Embedding(pneg_vocab_size, pneg_embed_size)
        self.field_rep_embed_size = field_embed_size+ppos_embed_size+pneg_embed_size
        self.encoder = Encoder(self.field_rep_embed_size, encoder_hidden_size, value_embed_size, verbose)
        self.decoder = Decoder(sent_embed_size, decoder_hidden_size, encoder_hidden_size, decoder_num_layer, sent_vocab_size, self.field_rep_embed_size, verbose)
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
        #print sent
        sent = self.sent_lookup(sent)
        #print sent
        encoder_initial_hidden = self.encoder.init_hidden(batch_size, self.encoder_hidden_size)
        if self.cuda_var:
            encoder_initial_hidden = encoder_initial_hidden.cuda()
        encoder_output, encoder_hidden = self.encoder.forward(input_d=input_d, input_z=input_z, hidden=encoder_initial_hidden, mask=value_mask)
        encoder_output = torch.stack(encoder_output, dim=0)
        encoder_hidden = (encoder_hidden[0].unsqueeze(0), encoder_hidden[1].unsqueeze(0))
        if self.verbose: print(encoder_output.size(), encoder_hidden[0].size(), encoder_hidden[1].size())
        # hidden = (table_hidden[0].view(-1, table_hidden[0].size(0)*table_hidden[0].size(2)).unsqueeze(0), table_hidden[1].view(-1, table_hidden[1].size(0)*table_hidden[1].size(2)).unsqueeze(0) )
        #print torch.stack(encoder_output, dim=0).shape # (111L, 32L, 500L)
        #print encoder_hidden.shape
        # encoder_output is list of all "hiddens" at each time step, do we need cell state too?
        decoder_output, decoder_hidden, attn_vectors = self.decoder.forward_biased_lstm(input=sent, hidden=encoder_hidden, encoder_hidden=encoder_output, input_z=input_z, mask=value_mask)
        return decoder_output, decoder_hidden

    def update_vectors(self):
        pass