import os
import torch
import re




class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Data(object):
    def __init__(self,table, sentence):
        self.box = table
        self.sent = sentence

    def add_data(self, table, sentence):
        self.box = table
        self.sent = sentence


class Corpus(object):
    def __init__(self, path, vocab_path, top_k, limit, verbose): # top_k is the number of sentences we would be generating
        self.field_vocab = Dictionary()
        self.word_vocab = Dictionary()


        self.field_ununk_vocab = Dictionary()
        self.word_ununk_vocab = Dictionary()

        self.pos_vocab = Dictionary()


        self.train_ppos = []
        self.train_ppos_len = []
        self.train_pneg = []
        self.train_pneg_len = []
        self.train_field = []
        self.train_field_len = []
        self.train_value = []
        self.train_value_len = []
        self.train_sent = []
        self.train_sent_len = []
        self.train_ununk_sent = []
        self.train_ununk_field = []
        self.train_ununk_value = []
        self.train_sent_mask = []
        self.train_value_mask = []

        self.test_ppos = []
        self.test_ppos_len = []
        self.test_pneg = []
        self.test_pneg_len = []
        self.test_field = []
        self.test_field_len = []
        self.test_value = []
        self.test_value_len = []
        self.test_sent = []
        self.test_sent_len = []
        self.test_ununk_sent = []
        self.test_ununk_field = []
        self.test_ununk_value = []
        self.test_sent_mask = []
        self.test_value_mask = []

        self.valid_ppos = []
        self.valid_ppos_len = []
        self.valid_pneg = []
        self.valid_pneg_len = []
        self.valid_field = []
        self.valid_field_len = []
        self.valid_value = []
        self.valid_value_len = []
        self.valid_sent = []
        self.valid_sent_len = []
        self.valid_ununk_sent = []
        self.valid_ununk_field = []
        self.valid_ununk_value = []
        self.valid_sent_mask = []
        self.valid_value_mask = []

        self.vocab = {"word_vocab": self.word_vocab, "field_vocab": self.field_vocab, "pos_vocab": self.pos_vocab, \
        "word_ununk_vocab": self.word_ununk_vocab, "field_ununk_vocab": self.field_ununk_vocab}
        self.verbose = verbose

        self.data_path = [['train/train.sent', 'train/train.nb', 'train/train.box'],
        ['test/test.sent', 'test/test.nb',  'test/test.box'],
        ['valid/valid.sent', 'valid/valid.nb', 'valid/valid.box']]

        self.populate_vocab(vocab_path, verbose)
        self.train_value, self.train_field,\
        self.train_ppos, self.train_pneg,\
        self.train_sent, self.train_ununk_sent, self.train_ununk_field,  \
        self.train_ununk_value  = self.new_populate_stores(path, self.data_path[0], top_k, limit, verbose)

        self.test_value, self.test_field, \
        self.test_ppos, self.test_pneg, \
        self.test_sent, self.test_ununk_sent, self.test_ununk_field,  \
        self.test_ununk_value = self.new_populate_stores(path, self.data_path[1], top_k, limit, verbose)

        self.valid_value, self.valid_field, \
        self.valid_ppos, self.valid_pneg, \
        self.valid_sent, self.valid_ununk_sent, self.valid_ununk_field,  \
        self.valid_ununk_value = self.new_populate_stores(path, self.data_path[2], top_k, limit, verbose)


        # Done: write a function here to get the frequencies too of each word, basically dict (word)-> frequency
        self.word2freq = {}
        sent_list = [self.train_ununk_sent, self.valid_ununk_sent, self.test_ununk_sent]
        for s_list in sent_list: # each s_list is a list of lists
            for sent in s_list:
                for word in sent:
                    if word not in self.word2freq:
                        self.word2freq[word]=0
                    self.word2freq[word]+=1
        self.word2freq[self.word_vocab.word2idx['UNK']]=1
        self.word2freq[self.word_vocab.word2idx['<eos>']]=1
        self.word2freq[self.word_vocab.word2idx['<sos>']]=1
        self.word2freq[self.word_vocab.word2idx['<pad>']]=1






    def create_data_dictionaries(self):
        self.train = {"value": self.train_value, "value_len": self.train_value_len,
                      "field": self.train_field, "field_len": self.train_field_len,
                      "ppos": self.train_ppos, "ppos_len": self.train_ppos_len,
                      "pneg": self.train_pneg, "pneg_len": self.train_pneg_len,
                      "sent": self.train_sent, "sent_len": self.train_sent_len,
                       "sent_ununk": self.train_ununk_sent, "field_ununk": self.train_ununk_field,
                      "value_ununk": self.train_ununk_value, "sent_mask": self.train_sent_mask,
                      "value_mask": self.train_value_mask}
        self.valid = {"value": self.valid_value, "value_len": self.valid_value_len,
                      "field": self.valid_field, 'field_len': self.valid_field_len,
                      "ppos": self.valid_ppos, "ppos_len": self.valid_ppos_len,
                      "pneg": self.valid_pneg,"pneg_len": self.valid_pneg_len,
                      "sent": self.valid_sent, "sent_len": self.valid_sent_len,
                      "sent_ununk": self.valid_ununk_sent, "field_ununk": self.valid_ununk_field,
                      "value_ununk": self.valid_ununk_value, "sent_mask": self.valid_sent_mask,
                      "value_mask": self.valid_value_mask}
        self.test = {"value": self.test_value, "value_len": self.test_value_len,
                     "field": self.test_field, 'field_len': self.test_field_len,
                     "ppos": self.test_ppos, "ppos_len": self.test_ppos_len,
                     "pneg": self.test_pneg, "pneg_len": self.test_pneg_len,
                     "sent": self.test_sent, "sent_len": self.test_sent_len,
                     "sent_ununk": self.test_ununk_sent, "field_ununk": self.test_ununk_field,
                     "value_ununk": self.test_ununk_value, "sent_mask": self.test_sent_mask,
                     "value_mask": self.test_value_mask}


    def populate_vocab(self, vocab_path, verbose):
        file = open(os.path.join(vocab_path, "word_vocab.txt"), "r")
        words = [line.split('\n')[0] for line in file]
        file = open(os.path.join(vocab_path, "field_vocab.txt"), "r")
        fields = [line.split('\n')[0] for line in file]
        self.word_vocab.add_word('<pad>')
        self.field_vocab.add_word('<pad>')
        self.pos_vocab.add_word('<pad>')

        self.field_ununk_vocab.add_word('<pad>') # for field label
        self.field_ununk_vocab.add_word('<sos>')
        self.field_ununk_vocab.add_word('<eos>')
        self.word_ununk_vocab.add_word('<pad>') # for sentence words
        self.word_ununk_vocab.add_word('<sos>')
        self.word_ununk_vocab.add_word('<eos>')


        self.word_vocab.add_word('<sos>')
        self.word_vocab.add_word('<eos>')
        self.word_vocab.add_word('UNK')
        self.field_vocab.add_word('UNK')

        for word in words:
            #print word.split('\t')
            word, freq = word.split()
            self.word_vocab.add_word(word)
        for field in fields:
            #print field.split('\t')
            field, freq = field.split('\t')
            self.field_vocab.add_word(field)


    def reverse_pos(self, temp_ppos):
        #print temp_ppos
        temp_pneg = []
        current = []
        for i in range(len(temp_ppos)):
            if temp_ppos[i]==1:
                temp_pneg+= current[::-1] # append whatever in buffer, in reverse fashion
                current = [] # start a new buffer
                current.append(temp_ppos[i]) # append in buffer
            else:
                current.append(temp_ppos[i]) # append in buffer

        temp_pneg+= current[::-1]
        return temp_pneg


    def new_populate_stores(self, path, data_path, top_k, limit, verbose):
        # handle the sentences # handle the nb # tokenize the appendings
        file = open(os.path.join(path, data_path[0]), "r")
        sentences = [line.split('\n')[0] for line in file]
        file = open(os.path.join(path, data_path[1]), "r")
        no_sentences = [int(line.split('\n')[0]) for line in file]
        sent = []
        sent_ununk  = []
        ppos = []
        pneg = []
        field = []
        field_ununk = []
        value = []
        value_ununk = []
        z = 0
        size = int(limit*len(no_sentences))
        # for s in range(len(no_sentences)):
        for s in range(size):
            current = sentences[z:z + no_sentences[s]]
            z = z + no_sentences[s]
            current = current[0:top_k]
            temp_sent = []
            temp_sent_ununk = []
            for c in current:
                c = ['<sos>'] + c.split(' ') + ['<eos>']
                for word in c:
                    if word in self.word_vocab.word2idx:
                        temp_sent.append(self.word_vocab.word2idx[word])
                    else:
                        temp_sent.append(self.word_vocab.word2idx['UNK'])

                    self.word_ununk_vocab.add_word(word)
                    temp_sent_ununk.append(self.word_ununk_vocab.word2idx[word])
            sent.append(temp_sent)
            sent_ununk.append(temp_sent_ununk)

        # handle the table and tokenize
        file = open(os.path.join(path, data_path[2]), "r")
        count = 0
        for line in file:
            if count == size:
              break
            count += 1
            temp_ppos = []
            temp_pneg = []
            temp_field = []
            temp_field_ununk = []
            temp_value_ununk = []
            temp_value = []
            pointers = []
            line = line.split('\n')[0].split('\t')
            j = 0
            field_prev = ""
            for l in line:  # address each part in the table for f, p+, p-, and value
                word = l.split(':')[0] # prefix
                field_value = l.split(':')[1] # word
                if '<none>' in field_value or field_value.strip()=='' or word.strip()=='':
                    continue

                word = l.split(':')[0].rsplit('_',1)[0] # field_name
                pos = l.split(':')[0].rsplit('_',1)[1]

                if word in self.field_vocab.word2idx:
                    temp_field.append(self.field_vocab.word2idx[word])
                else:
                    temp_field.append(self.field_vocab.word2idx['UNK'])

                self.field_ununk_vocab.add_word(word)
                temp_field_ununk.append(self.field_ununk_vocab.word2idx[word])


                if field_value in self.word_vocab.word2idx:

                    temp_value.append(self.word_vocab.word2idx[field_value])
                else:
                    temp_value.append(self.word_vocab.word2idx['UNK'])

                self.word_ununk_vocab.add_word(field_value)
                temp_value_ununk.append(self.word_ununk_vocab.word2idx[field_value])


                if re.search("[1-9]\d*$", pos):
                    field_id = int(pos)
                    if field_id<=30:
                        self.pos_vocab.add_word(field_id)
                        temp_ppos.append(self.pos_vocab.word2idx[field_id])
                    else:
                        self.pos_vocab.add_word(30)
                        temp_ppos.append(self.pos_vocab.word2idx[30])
                else:
                    self.pos_vocab.add_word(1)
                    temp_ppos.append(self.pos_vocab.word2idx[1])
                j+=1
                if j==100:
                    break

            temp_pneg = self.reverse_pos(temp_ppos)
            # TODO: call here to reverse it and redo the job for pneg
            value.append(temp_value)
            value_ununk.append(temp_value_ununk)

            field.append(temp_field)
            field_ununk.append(temp_field_ununk)

            ppos.append(temp_ppos)
            pneg.append(temp_pneg)
        return value, field, ppos, pneg, sent, sent_ununk, field_ununk, value_ununk
