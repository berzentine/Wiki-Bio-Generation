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
        tokens = 0
        k=0
        fp = open("./data/Wiki-Data/wikipedia-biography-dataset/train/train.sent.formatted1", "w")
        for t in self.train:
            k+=1
            fields = []
            field_values = {}
            content = t.box.split('\t')
            for c in content:
                parts = c.split(":")
                fields.append(parts[0])
                for p in parts[1:]:
                    if parts[0] in field_values:
                        field_values[parts[0]].append(p)
                    else:
                        field_values[parts[0]] = [p]
            sentences = []
            content = t.sent.split('\n')
            for c in content:
                new_words = []
                words = c.split(' ')
                for word in words:
                    flag = True
                    for field in fields:
                        if word in field_values[field]:
                            new_words.append(field)
                            flag = False
                            break
                    if flag:
                        new_words.append(word)
                sentences.append(" ".join(new_words))
            for sentence in sentences:
                fp.write(sentence+"\n")

        fp = open("./data/Wiki-Data/wikipedia-biography-dataset/test/test.sent.formatted1", "w")
        for t in self.test:
            k+=1
            fields = []
            field_values = {}
            content = t.box.split('\t')
            for c in content:
                parts = c.split(":")
                fields.append(parts[0])
                for p in parts[1:]:
                    if parts[0] in field_values:
                        field_values[parts[0]].append(p)
                    else:
                        field_values[parts[0]] = [p]
            sentences = []
            content = t.sent.split('\n')
            for c in content:
                new_words = []
                words = c.split(' ')
                for word in words:
                    flag = True
                    for field in fields:
                        if word in field_values[field]:
                            new_words.append(field)
                            flag = False
                            break
                    if flag:
                        new_words.append(word)
                sentences.append(" ".join(new_words))
            for sentence in sentences:
                fp.write(sentence+"\n")

        fp = open("./data/Wiki-Data/wikipedia-biography-dataset/valid/valid.sent.formatted1", "w")
        for t in self.train:
            k+=1
            fields = []
            field_values = {}
            content = t.box.split('\t')
            for c in content:
                parts = c.split(":")
                fields.append(parts[0])
                for p in parts[1:]:
                    if parts[0] in field_values:
                        field_values[parts[0]].append(p)
                    else:
                        field_values[parts[0]] = [p]
            sentences = []
            content = t.sent.split('\n')
            for c in content:
                new_words = []
                words = c.split(' ')
                for word in words:
                    flag = True
                    for field in fields:
                        if word in field_values[field]:
                            new_words.append(field)
                            flag = False
                            break
                    if flag:
                        new_words.append(word)
                sentences.append(" ".join(new_words))
            for sentence in sentences:
                fp.write(sentence+"\n")



    def build_data(self, trainsent, trainnb, testsent, testnb, valsent, valnb, traintab, testtab, valtab):
        k = 0
        for i in range(0,len(trainnb)):
            self.train.append(Data(traintab[i],'\n'.join(trainsent[k:k+int(trainnb[i].split()[0])])))
            k = k+int(trainnb[i].split()[0])
            #if i==11:
            #    break
        k = 0
        for i in range(0,len(testnb)):
            self.test.append(Data(testtab[i],'\n'.join(testsent[k:k+int(testnb[i].split()[0])])))
            k = k+int(testnb[i].split()[0])
            #if i==11:
            #    break
        k = 0
        for i in range(0,len(valnb)):
            self.val.append(Data(valtab[i],'\n'.join(valsent[k:k+int(valnb[i].split()[0])])))
            k = k+int(valnb[i].split()[0])
            #if i==11:
            #    break
        return self.train, self.val, self.test



corpus = Corpus('./data/Wiki-Data/wikipedia-biography-dataset/')
