import torch
import torch.nn as nn
from torch.autograd import Variable

from models.lstm_dual_attention_decoder_alignments_timestep import LSTMDualAttention


class Seq2SeqDualModelAlignTimestep(nn.Module):
    def __init__(self, sent_vocab_size, field_vocab_size, ppos_vocab_size, pneg_vocab_size, value_vocab_size, sent_embed_size, field_embed_size, \
                 value_embed_size, ppos_embed_size, pneg_embed_size, encoder_hidden_size, decoder_hidden_size, decoder_num_layer, verbose, cuda_var, x, pretrained = None, dropout=0.5):
        super(Seq2SeqDualModelAlignTimestep, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.drop = nn.Dropout(dropout)
        self.sent_lookup = nn.Embedding(sent_vocab_size, sent_embed_size)
        self.field_lookup = nn.Embedding(field_vocab_size, field_embed_size)
        self.ppos_lookup = nn.Embedding(ppos_vocab_size, ppos_embed_size)
        self.pneg_lookup = nn.Embedding(pneg_vocab_size, pneg_embed_size)
        self.field_rep_embed_size = field_embed_size+ppos_embed_size+pneg_embed_size
        #self.decoder = nn.LSTM(input_size=sent_embed_size, hidden_size=encoder_hidden_size, num_layers=1, bidirectional=False, batch_first=True)
        self.encoder = nn.LSTM(input_size=sent_embed_size+self.field_rep_embed_size, hidden_size=decoder_hidden_size//2, num_layers=1, bidirectional=True, batch_first=True, dropout=dropout)
        self.decoder = LSTMDualAttention(vocab_size = sent_vocab_size, input_size=sent_embed_size, field_rep_size=self.field_rep_embed_size, hidden_size=decoder_hidden_size, encoder_hidden_size=encoder_hidden_size, batch_first=True, dropout=dropout)
        self.linear_out = nn.Linear(encoder_hidden_size, sent_vocab_size)
        self.verbose = verbose
        self.cuda_var = cuda_var
        self.target_lookup = nn.Embedding(sent_vocab_size, 300)
        if pretrained is not None:
            self.target_lookup.weight.data.copy_(torch.from_numpy(pretrained))
            for param in self.target_lookup.parameters():
                param.requires_grad = False

        self.init_weights()
        #self.x = nn.Parameter(torch.zeros(1), requires_grad=True)



    def init_weights(self):
        torch.nn.init.xavier_uniform(self.sent_lookup.weight)
        torch.nn.init.xavier_uniform(self.field_lookup.weight)
        torch.nn.init.xavier_uniform(self.ppos_lookup.weight)
        torch.nn.init.xavier_uniform(self.pneg_lookup.weight)

    # converts the max attended index to a table word to lookup
    def position_idx_to_value(self, max_index, value): # max_index -> batch X sent_length
        indx2word = Variable(torch.LongTensor(max_index.size(0), max_index.size(1)))
        if self.cuda_var:
            indx2word = indx2word.cuda()
        for i in range(max_index.size(0)):
            for j in range(max_index.size(1)):
                indx2word[i,j] = value[i, int(max_index[i,j])]
        return indx2word


    def forward_with_attn(self, sent, value, field, ppos, pneg, batch_size, value_mask, align_prob, epsilon=0.001):

        input_d = self.sent_lookup(value)
        input_z = torch.cat((self.field_lookup(field), self.ppos_lookup(ppos), self.pneg_lookup(pneg)), 2)
        input = torch.cat((input_d,input_z), 2)
        input = self.drop(input)
        encoder_output, encoder_hidden = self.encoder(input, None)
        #encoder_hidden = None
        sent = self.sent_lookup(sent)
        encoder_hidden = (encoder_hidden[0].view(1, encoder_hidden[0].size(1), encoder_hidden[0].size(0)*encoder_hidden[0].size(2)), encoder_hidden[1].view(1, encoder_hidden[1].size(1), encoder_hidden[1].size(0)*encoder_hidden[1].size(2)))
        encoder_hidden = (encoder_hidden[0].squeeze(0),encoder_hidden[1].squeeze(0))
        decoder_output, decoder_hidden, attn, lamda = self.decoder.forward(sent, encoder_hidden, input_z, encoder_output) # lambda should be # 78* (32L, 1L),
        decoder_output = self.drop(decoder_output)
        decoder_output = self.linear_out(decoder_output)
        logsoftmax = nn.LogSoftmax(dim=2)
        decoder_output = logsoftmax(decoder_output)
        #print(len(attn), decoder_output.size(), attn[0].size()) # (78, (32L, 78L, 20003L), (32L, 100L))
        # stack the attention vector in the second dimension -> basically convert the list of 78 attn vectors to (32, 78,100 ) single matrix
        attn = torch.stack(attn, dim=1) # (32L, 78L, 100L)
        lamda = torch.stack(lamda, dim=1) # (32L, 78L, 1L)
        #dim(align_prob) = batch X vocab X table length # (32L, 20003L, 100L)
        #align_prob = Variable(torch.rand(attn.size(0), attn.size(2), decoder_output.size(2)))    # (32L, 20003L, 100L)
        if self.cuda_var:
            align_prob = align_prob.cuda()
        p_lex = torch.bmm(attn, align_prob) # do attn . align_prob' -> (32L, 78L, 20003L) same dimensions as decoder output
        #p_lex - logsoftmax(p_lex)
        p_mod = decoder_output
        #print(lamda.size())
        #print(p_lex.size())
        #print(torch.mul(lamda, p_lex).size())
        #print((lamda*p_lex).size())
        p_bias = lamda*p_lex + (1-lamda)*p_mod # (32L, 78L, 20003L)
        #print(p_bias)
        out_softmax = nn.LogSoftmax(dim=2)
        p_bias = out_softmax(p_bias)

        ########################
        # adding additional LOSS
        ########################
        # attn_pred should be a prediction of 32 X 78, basically max in last dimension to get max attended at word each time
        max_val, max_idx = torch.max(attn, 2) #-> 32 X 78
        # Replace max_index with word_index from the table, like attended position 5 replace it with word at position 5
        word_idx = self.position_idx_to_value(max_idx, value) # ->   32 X 78
        if self.cuda_var:
            word_idx = word_idx.cuda()
        attn_pred = self.target_lookup(word_idx) # Replace word_index with embeddings

        max_val, max_idx = torch.max(p_bias, 2) #-> # handle decoder predictions
        decoder_pred =  self.target_lookup(max_idx)
        #print attn_pred.size(), decoder_pred.size() -> same dimensions (32L, 78L, 400L)


        #print(p_bias)
        return p_bias, decoder_hidden, attn_pred, decoder_pred # should return the changed and weighted decoder output and not this output
        # should return decoder_output + LfAi + e


    def generate(self,value, value_len, field, ppos, pneg, batch_size, \
                 train, max_length, start_symbol, end_symbol, dictionary, unk_symbol, \
                 ununk_dictionary, value_ununk, value_mask, sent, align_prob):
        input_d = self.sent_lookup(value)
        self.field_lookup(field).size(), self.ppos_lookup(ppos).size(), self.ppos_lookup(pneg).size()
        input_z = torch.cat((self.field_lookup(field), self.ppos_lookup(ppos), self.pneg_lookup(pneg)), 2)

        input = torch.cat((input_d,input_z), 2)
        encoder_output, encoder_hidden = self.encoder(input, None)
        gen_seq = [[] for b in range(batch_size)]
        unk_rep_seq = [[] for b in range(batch_size)]
        attention_matrix = [[] for b in range(batch_size)]
        start_symbol =  Variable(torch.LongTensor(batch_size,1).fill_(start_symbol))
        if self.cuda_var:
            start_symbol = start_symbol.cuda()
        curr_input = self.sent_lookup(start_symbol) #-> (batch, 1L, embed size)

        encoder_hidden = (encoder_hidden[0].view(1, encoder_hidden[0].size(1), encoder_hidden[0].size(0)*encoder_hidden[0].size(2)), \
                          encoder_hidden[1].view(1, encoder_hidden[1].size(1), encoder_hidden[1].size(0)*encoder_hidden[1].size(2)))

        prev_hidden =  (encoder_hidden[0].squeeze(0),encoder_hidden[1].squeeze(0))

        for i in range(max_length):
            # decoder_output, prev_hidden, attn_vector = model.decoder.forward_biased_lstm(input=curr_input, hidden=prev_hidden, encoder_hidden=encoder_output, input_z=input_z, mask=value_mask)
            decoder_output, prev_hidden, attn_vector, lamda = self.decoder.forward(curr_input, prev_hidden, input_z, encoder_output)
            # Need to change here to include prob alignments and use learned lambda here
            # TODO: Need to change here to incorporate alignment prob
            decoder_output = self.linear_out(decoder_output)
            attn_vector = torch.stack(attn_vector, dim=1) # ->(32L, 1L, 100L)
            logsoftmax = nn.LogSoftmax(dim=2)
            decoder_output = logsoftmax(decoder_output)
            lamda = torch.stack(lamda, dim=1) # (32L, 78L, 1L)

            #dim(align_prob) = batch X vocab X table length
            #align_prob = Variable(torch.rand(attn_vector.size(0), attn_vector.size(2), decoder_output.size(2)))   # (32L, 20003L, 100L)
            if self.cuda_var:
                align_prob = align_prob.cuda()
            #print(align_prob.size(), attn_vector.size()) # -> ((batchsize, vocab, tab_length), (batchsize, seq_legth, tab_length)) = ((32L, 20003L, 100L), (32L, 1L, 100L))
            p_lex = torch.bmm(attn_vector, align_prob) # do attn . align_prob' -> (32L, 1L, 20003L) same dimensions as decoder output
            p_mod = decoder_output
            p_bias = lamda*p_lex + (1-lamda)*p_mod # (32L, 78L, 20003L
            #print(lamda)
            out_softmax = nn.LogSoftmax(dim=2)
            p_bias = out_softmax(p_bias)
            decoder_output = p_bias # -> (batch, 1L, vocab)
            #print gen_seq
            max_val, max_idx = torch.max(decoder_output, 2) #-> (batch, 1L), (batch, 1L)
            curr_input =  self.sent_lookup(max_idx) #-> (batch, 1L, embed size)

            #attention_matrix = None
            #attention_matrix.append(attn_vector[0][0,:value_len[0]].data.cpu().numpy())

            for b in range(batch_size):
                max_word_index = int(max_idx[b,0])
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
                attention_matrix[b].append(attn_vector[b][0,:value_len[b]].data.cpu().numpy())
        return gen_seq, unk_rep_seq,attention_matrix
