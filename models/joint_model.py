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

    def forward(self, sent, value, field, ppos, pneg, batch_size):
        input_d = self.value_lookup(value)
        input_z = torch.cat((self.field_lookup(field), self.ppos_lookup(ppos), self.pneg_lookup(pneg)), 2)
        #print sent
        sent = self.sent_lookup(sent)
        #print sent
        encoder_initial_hidden = self.encoder.init_hidden(batch_size, self.encoder_hidden_size)
        if self.cuda_var:
            encoder_initial_hidden = encoder_initial_hidden.cuda()
        encoder_output, encoder_hidden = self.encoder.forward(input_d=input_d, input_z=input_z, hidden=encoder_initial_hidden)
        encoder_output = torch.stack(encoder_output, dim=0)
        encoder_hidden = (encoder_hidden[0].unsqueeze(0), encoder_hidden[1].unsqueeze(0))
        if self.verbose: print(encoder_output.size(), encoder_hidden[0].size(), encoder_hidden[1].size())
        # hidden = (table_hidden[0].view(-1, table_hidden[0].size(0)*table_hidden[0].size(2)).unsqueeze(0), table_hidden[1].view(-1, table_hidden[1].size(0)*table_hidden[1].size(2)).unsqueeze(0) )
        #print torch.stack(encoder_output, dim=0).shape # (111L, 32L, 500L)
        #print encoder_hidden.shape
        # TODO: Fix from here: [encoder_hidden is concatenation of hidden and cell state]
        # encoder_output is list of all "hiddens" at each time step, do we need cell state too?
        decoder_output, decoder_hidden, attn_vectors = self.decoder.forward_biased_lstm(input=sent, hidden=encoder_hidden, encoder_hidden=torch.stack(encoder_output, dim=0), input_z=input_z)
        return decoder_output, decoder_hidden

    # TODO: should it be given as a batch? In which case how to handle the break condition in this method?


    def update_vectors(self):
        pass

    def generate_beam(self, value, field, ppos, pneg, batch_size, train, max_length, start_symbol, end_symbol, dictionary, beam, verbose):
        input_d = self.value_lookup(value)
        input_z = torch.cat((self.field_lookup(field), self.ppos_lookup(ppos), self.pneg_lookup(pneg)), 2)
        encoder_initial_hidden = self.encoder.init_hidden(batch_size, self.encoder_hidden_size)
        if self.cuda_var:
            encoder_initial_hidden = encoder_initial_hidden.cuda()
        encoder_output, encoder_hidden = self.encoder.forward(input_d=input_d, input_z=input_z, hidden=encoder_initial_hidden)
        encoder_output = torch.stack(encoder_output, dim=0)
        encoder_hidden = (encoder_hidden[0].unsqueeze(0), encoder_hidden[1].unsqueeze(0))
        gen_seq = []
        gen_seq.append('<sos>')
        start_symbol =  Variable(torch.LongTensor(1,1).fill_(start_symbol))
        if self.cuda_var:
            start_symbol = start_symbol.cuda()
        curr_input = self.sent_lookup(start_symbol) # TODO: change here to look and handle batches
        prev_hidden = encoder_hidden




        outputs, scores , hiddens, inputs, candidates = [], [], [], [], [[] for k in range(beam)]
        candidate_scores = [[] for k in range(beam)]
        for j in range(beam):
            candidates[j].append(dictionary.idx2word[int(start_symbol)])
            candidate_scores[j].append(0)

        decoder_output, prev_hidden = self.decoder.forward_plain(input=curr_input, hidden=prev_hidden, encoder_hidden=torch.stack(encoder_output, dim=0), input_z=input_z)
        #decoder_output, prev_hidden = self.decoder.forward(input=curr_input, hidden=prev_hidden, encoder_hidden=torch.stack(encoder_output, dim=0), input_z=input_z)
        decoder_output = torch.nn.functional.softmax(decoder_output, dim = 2)
        values, indices = torch.topk(decoder_output, beam, 2)
        for j in range(beam): # for step time = 0
            outputs.append(indices[0,0,j].squeeze().data[0]) # what was the otput during this state
            scores.append(torch.log(values[0,0,j]).squeeze().data[0]) # what was the score of otput during this state
            hiddens.append(prev_hidden) # what was the produced hidden state for otput during this state
            inputs.append(curr_input) # what was the input during this state
            #candidates[j].append(int(outputs[j])) # update candidate vectors too with a + " "
            candidates[j].append(dictionary.idx2word[int(outputs[j])])
            candidate_scores[j].append(scores[j])
        #print indices, 'start after sos'


        for i in range(max_length-1):
            # store k X K ones here for each jth exploration in outputs
            temp_scores, temp_hiddens , temp_inputs, temp_outputs = [], [], [], []
            for j in range(beam):
                # explore outputs[j]
                #print 'For index', outputs[j],
                curr_input = self.sent_lookup(Variable(torch.LongTensor(1,1).fill_(outputs[j])))
                # prev_hidden is an issue here
                decoder_output, prev_hidden = self.decoder.forward_plain(input=curr_input, hidden=hiddens[j], encoder_hidden=torch.stack(encoder_output, dim=0), input_z=input_z)
                #decoder_output, prev_hidden, attn_vector = self.decoder.forward(input=curr_input, hidden=hiddens[j], encoder_hidden=torch.stack(encoder_output, dim=0), input_z=input_z)
                decoder_output = torch.nn.functional.softmax(decoder_output, dim = 2)
                #print decoder_output
                values, indices = torch.topk(torch.log(decoder_output)+scores[j], beam, 2)
                #print values, indices, 'is the top k'

                for p in range(beam): # append to temp_scores and all temp vectors the top k of outputs of [j]
                    temp_outputs.append(indices[0,0,p].squeeze().data[0])
                    temp_scores.append(values[0,0,j].squeeze().data[0])
                    temp_hiddens.append(prev_hidden)
                    temp_inputs.append(outputs[j]) # issue is here
            #exit(0)
            #print '='*32
            # if explore all options
            # take top k and update actual vectors again
            if verbose:
                print ('This', len(temp_scores), 'should be beam*beam size')
            zipped = zip(temp_outputs, temp_scores, temp_hiddens, temp_inputs)
            zipped.sort(key = lambda t: t[1], reverse=True)
            #print len(zipped), type(zipped[0]) # 25 <type 'tuple'>
            outputs, scores , hiddens, inputs = [], [], [], []
            for j in range(beam):
                outputs.append(zipped[j][0])
                scores.append(zipped[j][1])
                hiddens.append(zipped[j][2])
                inputs.append(zipped[j][3])
                #candidates[j].append(int(outputs[j]))
                candidates[j].append(dictionary.idx2word[int(outputs[j])])
                candidate_scores[j].append(scores[j])

            #for j in range(beam): # update candidate vectors too with a + " "

        if verbose:
            print(candidates)
            print(candidate_scores)

        return candidates, candidate_scores


    def generate(self, value, value_len, field, ppos, pneg, batch_size, train, max_length, start_symbol, end_symbol, dictionary, unk_symbol, ununk_dictionary, value_ununk):
        input_d = self.value_lookup(value)
        input_z = torch.cat((self.field_lookup(field), self.ppos_lookup(ppos), self.pneg_lookup(pneg)), 2)
        encoder_initial_hidden = self.encoder.init_hidden(batch_size, self.encoder_hidden_size)
        if self.cuda_var:
            encoder_initial_hidden = encoder_initial_hidden.cuda()
        encoder_output, encoder_hidden = self.encoder.forward(input_d=input_d, input_z=input_z, hidden=encoder_initial_hidden)
        encoder_output = torch.stack(encoder_output, dim=0)
        encoder_hidden = (encoder_hidden[0].unsqueeze(0), encoder_hidden[1].unsqueeze(0))
        gen_seq = []
        unk_rep_seq = []
        #gen_seq.append('<sos>')
        #print start_symbol
        # hsould be a 1 X 1 long tensor
        start_symbol =  Variable(torch.LongTensor(1,1).fill_(start_symbol))
        if self.cuda_var:
            start_symbol = start_symbol.cuda()
        #print start_symbol
        curr_input = self.sent_lookup(start_symbol) # TODO: change here to look and handle batches
        prev_hidden = encoder_hidden
        #print 'start', curr_input.shape  ## start (1L, 1L, 400L)
        for i in range(max_length):
            decoder_output, prev_hidden, attn_vector = self.decoder.forward_biased_lstm(input=curr_input, hidden=prev_hidden, encoder_hidden=torch.stack(encoder_output, dim=0), input_z=input_z)
            #print 'decoder out', decoder_output.squeeze().shape ## decoder out (20003L,)
            #attn_vector = torch.stack(attn_vector, dim=0)
            max_val, max_idx = torch.max(decoder_output.squeeze(), 0)
            #print 'max index', int(max_idx)
            curr_input = self.sent_lookup(max_idx).unsqueeze(0)
            # TODO if max_idx is UNK then do what?
            #print 'new curr', curr_input.shape
            if int(max_idx) == unk_symbol:
                if self.cuda_var:
                    value_ununk = value_ununk.cuda()
                unk_max_val, unk_max_idx = torch.max(attn_vector[0][0,:value_len[0],0], 0)
                #print(type(unk_max_val), type(unk_max_idx))
                #print value_ununk[0]
                #print type(value_ununk[0])
                sub = value_ununk[0][unk_max_idx] # should be value_ununk
                word = ununk_dictionary.idx2word[int(sub)] # should be replaced from ununk dictionary word_ununk_vocab
                print("Unk got replaced with", word)
            else:
                word = dictionary.idx2word[int(max_idx)]
            gen_seq.append(dictionary.idx2word[int(max_idx)])
            unk_rep_seq.append(word)
            #print gen_seq
            if dictionary.idx2word[int(max_idx)] == '<eos>':
                #gen_seq.append('<eos>')
                #print('breaked', '='*32)
                break
        #print(gen_seq)
        return gen_seq, unk_rep_seq
