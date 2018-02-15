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

    def add_data(table, sentence):
        self.box = table
        self.sent = sentence


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()

        file = open(os.path.join(path, 'train/train.sent'), "r")
        trainsent = [line.split('\n')[0] for line in file]
        file = open(os.path.join(path, 'train/train.nb'), "r")
        trainnb = [ line for line in file]
        file = open(os.path.join(path, 'train/train.box'), "r")
        traintab = [ line.split('\n')[0] for line in file]

        file = open(os.path.join(path, 'test/test.sent'), "r")
        testsent = [ line.split('\n')[0] for line in file]
        file = open(os.path.join(path, 'test/test.nb'), "r")
        testnb = [ line for line in file]
        file = open(os.path.join(path, 'test/test.box'), "r")
        testtab = [ line.split('\n')[0] for line in file]

        file = open(os.path.join(path, 'valid/valid.sent'), "r")
        valsent = [ line.split('\n')[0] for line in file]
        file = open(os.path.join(path, 'valid/valid.box'), "r")
        valtab = [ line.split('\n')[0] for line in file]
        file = open(os.path.join(path, 'valid/valid.nb'), "r")
        valnb = [ line for line in file]

        self.train, self.val, self.test = self.build_data(trainsent, trainnb, testsent, testnb, valsent, valnb, traintab, testtab, valtab)
        # TODO: tokenize each of these and update here , should return updated dictionary, and tokenized text
        self.tokenize()


    def tokenize(self):
        self.dictionary



    def build_data(self, trainsent, trainnb, testsent, testnb, valsent, valnb, traintab, testtab, valtab):
        k = 0
        for i in range(0,len(trainnb)):
            self.train.append(Data(traintab[i],' '.join(trainsent[k:k+int(trainnb[i].split()[0])])))
            k = k+int(trainnb[i].split()[0])
        k = 0
        for i in range(0,len(testnb)):
            self.test.append(Data(testtab[i],' '.join(testsent[k:k+int(testnb[i].split()[0])])))
            k = k+int(testnb[i].split()[0])
        k = 0
        for i in range(0,len(valnb)):
            self.val.append(Data(valtab[i],' '.join(valsent[k:k+int(valnb[i].split()[0])])))
            k = k+int(valnb[i].split()[0])
        return self.train, self.val, self.test



        # for every line in the file
            # create a new object and add it to the file
        #return self.train, self.val, self.test
        # train is a list of objects with train[0].box being the table and train[0].sent being the sentence
