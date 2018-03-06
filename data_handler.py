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
    def __init__(self, path, top_k): # top_k is the number of sentences we would be generating

        self.train_ppos_dict = Dictionary()
        self.train_sent_dict = Dictionary()
        self.train_pneg_dict = Dictionary()
        self.train_field_dict = Dictionary()
        self.train_value_dict = Dictionary()
        self.train_ppos = []
        self.train_pneg = []
        self.train_field = []
        self.train_value = []
        self.train_sent = []

        self.test_ppos_dict = Dictionary()
        self.test_pneg_dict = Dictionary()
        self.test_field_dict = Dictionary()
        self.test_value_dict = Dictionary()
        self.test_sent_dict = Dictionary()
        self.test_sent = []
        self.test_ppos = []
        self.test_pneg = []
        self.test_field = []
        self.test_value = []

        self.valid_ppos_dict = Dictionary()
        self.valid_pneg_dict = Dictionary()
        self.valid_field_dict = Dictionary()
        self.valid_value_dict = Dictionary()
        self.valid_sent_dict = Dictionary()
        self.valid_sent = []
        self.valid_ppos = []
        self.valid_pneg = []
        self.valid_field = []
        self.valid_value = []

        data = [['train/train.sent', 'train/train.nb', 'train/train.box'], \
        ['test/test.sent', 'test/test.nb',  'test/test.box'], \
        ['valid/valid.sent', 'valid/valid.nb', 'valid/valid.box']]

        data_store = [[self.train_value, self.train_field , self.train_ppos, self.train_pneg, self.train_sent],\
        [self.test_value, self.test_field , self.test_ppos, self.test_pneg, self.test_sent],\
        [self.valid_value, self.valid_field , self.valid_ppos, self.valid_pneg, self.valid_sent]]

        data_dict = [[self.train_value_dict, self.train_field_dict , self.train_ppos_dict, self.train_pneg_dict, self.train_sent_dict],\
        [self.test_value_dict, self.test_field_dict , self.test_ppos_dict, self.test_pneg_dict, self.test_sent_dict],\
        [self.valid_value_dict, self.valid_field_dict , self.valid_ppos_dict, self.valid_pneg_dict, self.valid_sent_dict]]

        for i in range(0,3):
            if i ==1: print 'Done loading train data'
            if i ==2: print 'Done loading test data'
            # handle the sentences # handle the nb # tokenize the appendings
            file = open(os.path.join(path, data[i][0]), "r")
            sentences = [line.split('\n')[0] for line in file]
            file = open(os.path.join(path, data[i][1]), "r")
            no_sentences = [int(line.split('\n')[0]) for line in file]
            data_dict[i][4].add_word('<pad>')
            z = 0
            for s in range(len(no_sentences)):
                current = sentences[z:z+no_sentences[s]]
                z=z+no_sentences[s]
                current =  current[0:top_k]
                temp_sent = []
                for c in current:
                    c = ['<sos>']  + c.split(' ') + ['<eos>']
                    #print c
                    for word in c:
                        data_dict[i][4].add_word(word)
                        temp_sent.append(data_dict[i][4].word2idx[word])
                    #print temp_sent
                    #print '*'*32
                data_store[i][4].append(temp_sent)

            # handle the table and tokenize
            file = open(os.path.join(path, data[i][2]), "r")
            for line in file:
                data_dict[i][0].add_word('<pad>')
                data_dict[i][1].add_word('<pad>')
                data_dict[i][2].add_word('<pad>')
                data_dict[i][3].add_word('<pad>')

                temp_ppos = []
                temp_pneg = []
                temp_field = []
                temp_value = []
                line = line.split('\n')[0].split('\t')
                j = 0
                for l in line: # address each part in the table for f, p+, p-, and value
                    data_dict[i][1].add_word(l.split(':')[0])
                    temp_field.append(data_dict[i][1].word2idx[l.split(':')[0]])

                    data_dict[i][0].add_word(l.split(':')[1])
                    temp_value.append(data_dict[i][0].word2idx[l.split(':')[1]])

                    data_dict[i][2].add_word(j)
                    temp_ppos.append(data_dict[i][2].word2idx[j])

                    data_dict[i][3].add_word(len(line)-j-1)
                    temp_pneg.append(data_dict[i][3].word2idx[len(line)-j-1])
                    j+=1

                data_store[i][0].append(temp_value)
                data_store[i][1].append(temp_field)
                data_store[i][2].append(temp_ppos)
                data_store[i][3].append(temp_pneg)
            #break
        print 'Done loading val. data'
