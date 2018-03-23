import os
import torch



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
        self.vocab = {"word_vocab": self.word_vocab, "field_vocab": self.field_vocab, "pos_vocab": self.pos_vocab}
        self.verbose = verbose

        self.data_path = [['train/train.sent', 'train/train.nb', 'train/train.box'],
        ['test/test.sent', 'test/test.nb',  'test/test.box'],
        ['valid/valid.sent', 'valid/valid.nb', 'valid/valid.box']]

        self.populate_vocab(vocab_path, verbose)
        self.train_value, self.train_field,\
        self.train_ppos, self.train_pneg,\
        self.train_sent = self.new_populate_stores(path, self.data_path[0], top_k, limit, verbose)

        self.test_value, self.test_field, \
        self.test_ppos, self.test_pneg, \
        self.test_sent = self.new_populate_stores(path, self.data_path[1], top_k, limit, verbose)

        self.valid_value, self.valid_field, \
        self.valid_ppos, self.valid_pneg, \
        self.valid_sent = self.new_populate_stores(path, self.data_path[2], top_k, limit, verbose)


    def create_data_dictionaries(self):
        self.train = {"value": self.train_value, "value_len": self.train_value_len,
                      "field": self.train_field, "field_len": self.train_field_len,
                      "ppos": self.train_ppos, "ppos_len": self.train_ppos_len,
                      "pneg": self.train_pneg, "pneg_len": self.train_pneg_len,
                      "sent": self.train_sent, "sent_len": self.train_sent_len}
        self.valid = {"value": self.valid_value, "value_len": self.valid_value_len,
                      "field": self.valid_field, 'field_len': self.valid_field_len,
                      "ppos": self.valid_ppos, "ppos_len": self.valid_ppos_len,
                      "pneg": self.valid_pneg,"pneg_len": self.valid_pneg_len,
                      "sent": self.valid_sent, "sent_len": self.valid_sent_len}
        self.test = {"value": self.test_value, "value_len": self.test_value_len,
                     "field": self.test_field, 'field_len': self.test_field_len,
                     "ppos": self.test_ppos, "ppos_len": self.test_ppos_len,
                     "pneg": self.test_pneg, "pneg_len": self.test_pneg_len,
                     "sent": self.test_sent, "sent_len": self.test_sent_len}





    def populate_vocab(self, vocab_path, verbose):
        file = open(os.path.join(vocab_path, "word_vocab.txt"), "r")
        words = [line.split('\n')[0] for line in file]
        file = open(os.path.join(vocab_path, "field_vocab.txt"), "r")
        fields = [line.split('\n')[0] for line in file]
        self.word_vocab.add_word('<pad>')
        self.field_vocab.add_word('<pad>')
        self.pos_vocab.add_word('<pad>')
        self.word_vocab.add_word('<sos>')
        self.word_vocab.add_word('<eos>')
        self.word_vocab.add_word('UNK')
        self.field_vocab.add_word('UNK')
        for word in words:
            word, freq = word.split('\t')
            self.word_vocab.add_word(word)
        for field in fields:
            field, freq = field.split('\t')
            self.field_vocab.add_word(field)

    def reverse_pos(line, pointers, temp_ppos):
        print temp_ppos
        print pointers
        return []
    def new_populate_stores(self, path, data_path, top_k, limit, verbose):
        # handle the sentences # handle the nb # tokenize the appendings
        file = open(os.path.join(path, data_path[0]), "r")
        sentences = [line.split('\n')[0] for line in file]
        file = open(os.path.join(path, data_path[1]), "r")
        no_sentences = [int(line.split('\n')[0]) for line in file]
        sent = []
        ppos = []
        pneg = []
        field = []
        value = []
        z = 0
        size = int(limit*len(no_sentences))
        # for s in range(len(no_sentences)):
        for s in range(size):
            current = sentences[z:z + no_sentences[s]]
            z = z + no_sentences[s]
            current = current[0:top_k]
            temp_sent = []
            for c in current:
                c = ['<sos>'] + c.split(' ') + ['<eos>']
                for word in c:
                    if word in self.word_vocab.word2idx:
                        temp_sent.append(self.word_vocab.word2idx[word])
                    else:
                        temp_sent.append(self.word_vocab.word2idx['UNK'])
            sent.append(temp_sent)

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
            temp_value = []
            pointers = []
            line = line.split('\n')[0].split('\t')
            j = 0
            field_prev = ""
            for l in line:  # address each part in the table for f, p+, p-, and value
                word = l.split(':')[0].rsplit('_',1)[0] # field_name
                if field_prev!=word:
                    field_prev = word
                    pointers.append(j)
                j += 1
                # TODO: handle <none> for the split case, would give an error
                pos = l.split(':')[0].rsplit('_',1)[1]
                if word in self.field_vocab.word2idx:
                    temp_field.append(self.field_vocab.word2idx[word])
                else:
                    temp_field.append(self.field_vocab.word2idx['UNK'])

                word = l.split(':')[1]
                if word in self.word_vocab.word2idx:
                    temp_value.append(self.word_vocab.word2idx[word])
                else:
                    temp_value.append(self.word_vocab.word2idx['UNK'])
                if int(pos)<30:
                    self.pos_vocab.add_word(pos)
                    temp_ppos.append(self.pos_vocab.word2idx[pos])
                else:
                    temp_ppos.append(self.pos_vocab.word2idx['30'])


                """if j>30:
                    self.pos_vocab.add_word(30)
                    temp_ppos.append(self.pos_vocab.word2idx[30])
                else:
                    self.pos_vocab.add_word(j)
                    temp_ppos.append(self.pos_vocab.word2idx[j])

                if (len(line) - j - 1)>30:
                    self.pos_vocab.add_word(30)
                    temp_pneg.append(self.pos_vocab.word2idx[30])
                else:
                    self.pos_vocab.add_word(len(line) - j - 1)
                    temp_pneg.append(self.pos_vocab.word2idx[len(line) - j - 1])
                j += 1"""
            temp_pneg = reverse_pos(line, pointers, temp_ppos)
            # TODO: call here to reverse it and redo the job for pneg
            value.append(temp_value)
            field.append(temp_field)
            ppos.append(temp_ppos)
            pneg.append(temp_pneg)
        return value, field, ppos, pneg, sent
