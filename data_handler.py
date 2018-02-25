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
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = []
        self.val = []
        self.test = []
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
        self.tokenize_common()


    # Function to have a common dictionary for tables and biography texts
    def tokenize_common(self):
        self.dictionary.add_word('<pad>')
        # Add words to the dictionary
        tokens = 0
        for t in self.train:
            content = t.box.split('\t')
            for c in content:
                parts = c.split(":")
                for p in parts:
                    tokens += 1
                    self.dictionary.add_word(p)
            content = t.sent.split('\n')
            for c in content:
                words = c.split(' ') + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        for t in self.val:
            content = t.box.split('\t')
            for c in content:
                parts = c.split(":")
                for p in parts:
                    tokens += 1
                    self.dictionary.add_word(p)
            content = t.sent.split('\n')
            for c in content:
                words = c.split(' ') + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        for t in self.test:
            content = t.box.split('\t')
            for c in content:
                parts = c.split(":")
                for p in parts:
                    tokens += 1
                    self.dictionary.add_word(p)
            content = t.sent.split('\n')
            for c in content:
                words = c.split(' ') + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file
        for t in self.train:
            content = t.box.split('\t')
            for c in range(0,len(content)):
                parts = content[c].split(":")
                for p in range(0,len(parts)):
                    parts[p] = str(self.dictionary.word2idx[parts[p]])
                content[c] = ':'.join(parts)
            t.box = '\t'.join(content)

            content = t.sent.split('\n')
            for c in range(0,len(content)):
                words = content[c].split(' ') + ['<eos>']
                for word in range(0,len(words)):
                    words[word] = str(self.dictionary.word2idx[words[word]])
                content[c] = ' '.join(words)
            t.sent = '\n'.join(content)

        for t in self.val:
            content = t.box.split('\t')
            for c in range(0,len(content)):
                parts = content[c].split(":")
                for p in range(0,len(parts)):
                    parts[p] = str(self.dictionary.word2idx[parts[p]])
                content[c] = ':'.join(parts)
            t.box = '\t'.join(content)

            content = t.sent.split('\n')
            for c in range(0,len(content)):
                words = content[c].split(' ') + ['<eos>']
                for word in range(0,len(words)):
                    words[word] = str(self.dictionary.word2idx[words[word]])
                content[c] = ' '.join(words)
            t.sent = '\n'.join(content)

        for t in self.test:
            content = t.box.split('\t')
            for c in range(0,len(content)):
                parts = content[c].split(":")
                for p in range(0,len(parts)):
                    parts[p] = str(self.dictionary.word2idx[parts[p]])
                content[c] = ':'.join(parts)
            t.box = '\t'.join(content)

            content = t.sent.split('\n')
            for c in range(0,len(content)):
                words = content[c].split(' ') + ['<eos>']
                for word in range(0,len(words)):
                    words[word] = str(self.dictionary.word2idx[words[word]])
                content[c] = ' '.join(words)
            t.sent = '\n'.join(content)



    def build_data(self, trainsent, trainnb, testsent, testnb, valsent, valnb, traintab, testtab, valtab):
        k = 0
        for i in range(0,len(trainnb)):
            self.train.append(Data(traintab[i],'\n'.join(trainsent[k:k+int(trainnb[i].split()[0])])))
            k = k+int(trainnb[i].split()[0])
            if i==11:
                break
        k = 0
        for i in range(0,len(testnb)):
            self.test.append(Data(testtab[i],'\n'.join(testsent[k:k+int(testnb[i].split()[0])])))
            k = k+int(testnb[i].split()[0])
            if i==11:
                break
        k = 0
        for i in range(0,len(valnb)):
            self.val.append(Data(valtab[i],'\n'.join(valsent[k:k+int(valnb[i].split()[0])])))
            k = k+int(valnb[i].split()[0])
            if i==11:
                break
        return self.train, self.val, self.test
