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
    def __init__(self, path, top_k, verbose): # top_k is the number of sentences we would be generating

        self.train_ppos_dict = Dictionary()
        self.train_sent_dict = Dictionary()
        self.train_pneg_dict = Dictionary()
        self.train_field_dict = Dictionary()
        self.train_value_dict = Dictionary()
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
        self.train = {"value": self.train_value, "value_dict": self.train_value_dict, "value_len": self.train_value_len, \
                      "field": self.train_field, "field_dict": self.train_field_dict, 'field_len': self.train_field_len, \
                      "ppos": self.train_ppos, "ppos_dict": self.train_ppos_dict, "ppos_len": self.train_ppos_len, \
                      "pneg": self.train_pneg, "pneg_dict": self.train_pneg_dict, "pneg_len": self.train_pneg_len, \
                      "sent": self.train_sent, "sent_dict": self.train_sent_dict, "sent_len": self.train_sent_len}

        self.test_ppos_dict = Dictionary()
        self.test_pneg_dict = Dictionary()
        self.test_field_dict = Dictionary()
        self.test_value_dict = Dictionary()
        self.test_sent_dict = Dictionary()
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
        self.test = {"value": self.test_value, "value_dict": self.test_value_dict, "value_len": self.test_value_len, \
                     "field": self.test_field, "field_dict": self.test_field_dict, 'field_len': self.test_field_len, \
                     "ppos": self.test_ppos, "ppos_dict": self.test_ppos_dict, "ppos_len": self.test_ppos_len, \
                     "pneg": self.test_pneg, "pneg_dict": self.test_pneg_dict, "pneg_len": self.test_pneg_len, \
                     "sent": self.test_sent, "sent_dict": self.test_sent_dict, "sent_len": self.test_sent_len}

        self.valid_ppos_dict = Dictionary()
        self.valid_pneg_dict = Dictionary()
        self.valid_field_dict = Dictionary()
        self.valid_value_dict = Dictionary()
        self.valid_sent_dict = Dictionary()
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
        self.valid = {"value": self.valid_value, "value_dict": self.valid_value_dict, "value_len": self.valid_value_len, \
                      "field": self.valid_field, "field_dict": self.valid_field_dict, 'field_len': self.valid_field_len, \
                      "ppos": self.valid_ppos, "ppos_dict": self.valid_ppos_dict, "ppos_len": self.valid_ppos_len, \
                      "pneg": self.valid_pneg, "pneg_dict": self.valid_pneg_dict, "pneg_len": self.valid_pneg_len, \
                      "sent": self.valid_sent, "sent_dict": self.valid_sent_dict, "sent_len": self.valid_sent_len}
        self.verbose = verbose

        self.data_path = [['train/train.sent', 'train/train.nb', 'train/train.box'], \
                          ['test/test.sent', 'test/test.nb',  'test/test.box'], \
                          ['valid/valid.sent', 'valid/valid.nb', 'valid/valid.box']]

        self.train_value_dict, self.train_value, self.train_field_dict, self.train_field, \
        self.train_ppos_dict, self.train_ppos, self.train_pneg_dict, self.train_pneg, \
        self.train_sent_dict, self.train_sent = self.new_populate_stores(path, self.data_path[0], top_k, verbose)

        self.test_value_dict, self.test_value, self.test_field_dict, self.test_field, \
        self.test_ppos_dict, self.test_ppos, self.test_pneg_dict, self.test_pneg, \
        self.test_sent_dict, self.test_sent = self.new_populate_stores(path, self.data_path[1], top_k, verbose)

        self.valid_value_dict, self.valid_value, self.valid_field_dict, self.valid_field, \
        self.valid_ppos_dict, self.valid_ppos, self.valid_pneg_dict, self.valid_pneg, \
        self.valid_sent_dict, self.valid_sent = self.new_populate_stores(path, self.data_path[2], top_k, verbose)

    def new_populate_stores(self, path, data_path, top_k, verbose):
        # handle the sentences # handle the nb # tokenize the appendings
        file = open(os.path.join(path, data_path[0]), "r")
        sentences = [line.split('\n')[0] for line in file]
        file = open(os.path.join(path, data_path[1]), "r")
        no_sentences = [int(line.split('\n')[0]) for line in file]
        sent_dict = Dictionary()
        value_dict = Dictionary()
        field_dict = Dictionary()
        ppos_dict = Dictionary()
        pneg_dict = Dictionary()
        sent = []
        ppos = []
        pneg = []
        field = []
        value = []
        sent_dict.add_word('<pad>')
        z = 0
        for s in range(len(no_sentences)):
            current = sentences[z:z + no_sentences[s]]
            z = z + no_sentences[s]
            current = current[0:top_k]
            temp_sent = []
            for c in current:
                c = ['<sos>'] + c.split(' ') + ['<eos>']
                # print c
                for word in c:
                    sent_dict.add_word(word)
                    temp_sent.append(sent_dict.word2idx[word])
                    # print temp_sent
                    # print '*'*32
            sent.append(temp_sent)

        # handle the table and tokenize
        value_dict.add_word('<pad>')
        field_dict.add_word('<pad>')
        ppos_dict.add_word('<pad>')
        pneg_dict.add_word('<pad>')
        file = open(os.path.join(path, data_path[2]), "r")
        for line in file:
            temp_ppos = []
            temp_pneg = []
            temp_field = []
            temp_value = []
            line = line.split('\n')[0].split('\t')
            j = 0
            for l in line:  # address each part in the table for f, p+, p-, and value
                field_dict.add_word(l.split(':')[0])
                temp_field.append(field_dict.word2idx[l.split(':')[0]])

                value_dict.add_word(l.split(':')[1])
                temp_value.append(value_dict.word2idx[l.split(':')[1]])

                ppos_dict.add_word(j)
                temp_ppos.append(ppos_dict.word2idx[j])

                pneg_dict.add_word(len(line) - j - 1)
                temp_pneg.append(pneg_dict.word2idx[len(line) - j - 1])
                j += 1

            value.append(temp_value)
            field.append(temp_field)
            ppos.append(temp_ppos)
            pneg.append(temp_pneg)
        return value_dict, value, field_dict, field, ppos_dict, ppos, pneg_dict, pneg, sent_dict, sent