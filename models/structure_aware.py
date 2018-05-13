import torch
import torch.nn as nn
from torch.autograd import Variable

from models.lstm_dual_attention_decoder import LSTMDualAttention
from older_models.encoder import Encoder

class Seq2SeqDualModelSOTA(nn.Module):
    def __init__(self, sent_vocab_size, field_vocab_size, ppos_vocab_size, pneg_vocab_size, value_vocab_size, sent_embed_size, field_embed_size, \
                 value_embed_size, ppos_embed_size, pneg_embed_size, encoder_hidden_size, decoder_hidden_size, decoder_num_layer, verbose, cuda_var, x):
        super(Seq2SeqDualModelSOTA, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.sent_lookup = nn.Embedding(sent_vocab_size, sent_embed_size)
        self.field_lookup = nn.Embedding(field_vocab_size, field_embed_size)
        self.ppos_lookup = nn.Embedding(ppos_vocab_size, ppos_embed_size)
        self.pneg_lookup = nn.Embedding(pneg_vocab_size, pneg_embed_size)
        self.field_rep_embed_size = field_embed_size+ppos_embed_size+pneg_embed_size
        #self.decoder = nn.LSTM(input_size=sent_embed_size, hidden_size=encoder_hidden_size, num_layers=1, bidirectional=False, batch_first=True)
        self.encoder = Encoder(self.field_rep_embed_size, encoder_hidden_size, value_embed_size, verbose)
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


    def forward_with_attn(self, sent, value, field, ppos, pneg, batch_size, value_mask):

        input_d = self.sent_lookup(value)
        input_z = torch.cat((self.field_lookup(field), self.ppos_lookup(ppos), self.pneg_lookup(pneg)), 2)
        #input = torch.cat((input_d,input_z), 2)
        #encoder_output, encoder_hidden = self.encoder(input, None)
        #encoder_hidden = None
        encoder_initial_hidden = self.encoder.init_hidden(batch_size, self.encoder_hidden_size)
        if self.cuda_var:
            encoder_initial_hidden = encoder_initial_hidden.cuda()
        encoder_output, encoder_hidden = self.encoder.forward_test(input_d=input_d, input_z=input_z, hidden=encoder_initial_hidden)
        #encoder_hidden = None
        sent = self.sent_lookup(sent)
        #encoder_hidden = (encoder_hidden[0].view(1, encoder_hidden[0].size(1), encoder_hidden[0].size(0)*encoder_hidden[0].size(2)), encoder_hidden[1].view(1, encoder_hidden[1].size(1), encoder_hidden[1].size(0)*encoder_hidden[1].size(2)))
        #encoder_hidden = (encoder_hidden[0].squeeze(0),encoder_hidden[1].squeeze(0))
        encoder_output = torch.stack(encoder_output, dim=1)
        # issue is here , decoder expected batch X hidden
        #encoder_hidden = (encoder_hidden[0].unsqueeze(0), encoder_hidden[1].unsqueeze(0))
        decoder_output, decoder_hidden, attn = self.decoder.forward(sent, encoder_hidden, input_z, encoder_output)
        #decoder_output, decoder_hidden, attn_vectors = self.decoder.forward_biased_lstm(input=sent, hidden=encoder_hidden, encoder_hidden=encoder_output, input_z=input_z, mask=value_mask)
        decoder_output = self.linear_out(decoder_output)
        logsoftmax = nn.LogSoftmax(dim=2)
        decoder_output = logsoftmax(decoder_output)
        return decoder_output, decoder_hidden

        """sent = self.sent_lookup(sent)
        encoder_hidden = (encoder_hidden[0].view(1, encoder_hidden[0].size(1), encoder_hidden[0].size(0)*encoder_hidden[0].size(2)), encoder_hidden[1].view(1, encoder_hidden[1].size(1), encoder_hidden[1].size(0)*encoder_hidden[1].size(2)))
        encoder_hidden = (encoder_hidden[0].squeeze(0),encoder_hidden[1].squeeze(0))
        decoder_output, decoder_hidden, attn = self.decoder.forward(sent, encoder_hidden, input_z, encoder_output)
        decoder_output = self.linear_out(decoder_output)
        logsoftmax = nn.LogSoftmax(dim=2)
        decoder_output = logsoftmax(decoder_output)

        return decoder_output, decoder_hidden # should return the changed and weighted decoder output and not this output
        # should return decoder_output + LfAi + e"""

    def generate(self, value, value_len, field, ppos, pneg, batch_size, \
                 train, max_length, start_symbol, end_symbol, dictionary, unk_symbol, \
                 ununk_dictionary, value_ununk, value_mask, sent):
        input_d = self.sent_lookup(value)
        input_z = torch.cat((self.field_lookup(field), self.ppos_lookup(ppos), self.pneg_lookup(pneg)), 2)
        #input = torch.cat((input_d,input_z), 2)
        if self.cuda_var:
            encoder_initial_hidden = encoder_initial_hidden.cuda()
        encoder_output, encoder_hidden = self.encoder.forward_test(input_d=input_d, input_z=input_z, hidden=encoder_initial_hidden)
        encoder_output = torch.stack(encoder_output, dim=1)

        #encoder_output, encoder_hidden = self.encoder(input, None)
        gen_seq = [[] for b in range(batch_size)]
        unk_rep_seq = [[] for b in range(batch_size)]
        attention_matrix = [[] for b in range(batch_size)]
        start_symbol =  Variable(torch.LongTensor(1,1).fill_(start_symbol))
        if self.cuda_var:
            start_symbol = start_symbol.cuda()
        curr_input = self.sent_lookup(start_symbol) # TODO: change here to look and handle batches
        # print curr_input.shape()
        encoder_hidden = (encoder_hidden[0].view(1, encoder_hidden[0].size(1), encoder_hidden[0].size(0)*encoder_hidden[0].size(2)), encoder_hidden[1].view(1, encoder_hidden[1].size(1), encoder_hidden[1].size(0)*encoder_hidden[1].size(2)))
        # encoder_hidden = (encoder_hidden[0].squeeze(0),encoder_hidden[1].squeeze(0))
        prev_hidden =  (encoder_hidden[0].squeeze(0),encoder_hidden[1].squeeze(0))
        for i in range(max_length):
            decoder_output, prev_hidden, attn_vector = self.decoder.forward(curr_input, prev_hidden, input_z, encoder_output)
            # decoder_output, prev_hidden, attn_vector = model.decoder.forward_biased_lstm(input=curr_input, hidden=prev_hidden, encoder_hidden=encoder_output, input_z=input_z, mask=value_mask)
            #decoder_output, prev_hidden, attn_vector = self.decoder.forward(curr_input, prev_hidden, input_z, encoder_output)
            decoder_output = self.linear_out(decoder_output)
            attn_vector = torch.stack(attn_vector, dim=1)
            max_val, max_idx = torch.max(decoder_output, 2) #-> (batch, 1L), (batch, 1L)
            curr_input =  self.sent_lookup(max_idx) #-> (batch, 1L, embed size)

            for b in range(batch_size):
                max_word_index = int(max_idx[b, 0])
                if max_word_index == unk_symbol:
                    if self.cuda_var:
                        value_ununk = value_ununk.cuda()
                    # TODO: Double check this is correct
                    unk_max_val, unk_max_idx = torch.max(attn_vector[b][0,:value_len[b]], 0)
                    sub = value_ununk[b][unk_max_idx] # should be value_ununk
                    word = ununk_dictionary.idx2word[int(sub)] # should be replaced from ununk dictionary word_ununk_vocab
                    print("Unk got replaced with", word)
                else:
                    word = dictionary.idx2word[int(max_word_index)]
                    #print ('Ununk ', word)
                gen_seq[b].append(dictionary.idx2word[max_word_index])
                unk_rep_seq[b].append(word)
                # TODO: Double check this is correct
        return gen_seq, unk_rep_seq
