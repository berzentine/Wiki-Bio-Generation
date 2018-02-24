import torch
import numpy as np

def batchify(corpus):
    generate = 1 # generates only the top k sentence of the biography
    train_box = [] # List of tokensized sequences
    train_sent = []
    val_box = []
    val_sent = []
    test_box = []
    test_sent = []
    for example in corpus.train:
        temp = []
        for index in example.box.split('\t'):
            for i in index.split(':'):
                temp.append(i)
        train_box.append(temp)

        temp = []
        for index in range(0,len(example.sent.split('\n'))):
            if index<=generate-1:
                for i in example.sent.split('\n')[index].split():
                    temp.append(i)
        train_sent.append(temp)

    for example in corpus.val:
        temp = []
        for index in example.box.split('\t'):
            for i in index.split(':'):
                temp.append(i)
        val_box.append(temp)

        temp = []
        for index in range(0,len(example.sent.split('\n'))):
            if index<=generate-1:
                for i in example.sent.split('\n')[index].split():
                    temp.append(i)
        val_sent.append(temp)

    for example in corpus.test:
        temp = []
        for index in example.box.split('\t'):
            for i in index.split(':'):
                temp.append(i)
        test_box.append(temp)

        temp = []
        for index in range(0,len(example.sent.split('\n'))):
            if index<=generate-1:
                for i in example.sent.split('\n')[index].split():
                    temp.append(i)
        test_sent.append(temp)

    return train_box, train_sent, val_box, val_sent, test_box, test_sent
def pad_collate(datum,batchsize):
    train =  sorted(zip(datum[1],datum[0]), key=lambda pair: len(pair[0]))
    train_box = []
    train_sent = []
    for i in range(0,len(train),batchsize):
        batch = train[i:i+batchsize]
        max_sent = len(batch[-1][0])
        max_box = -100
        for b in batch:
            if max_box<len(b[1]):
                max_box=len(b[1])
        temp_box = []
        temp_sent = []
        for b in batch:
            while(len(b[0])<max_sent):
                b[0].append(0)
            b= ([int(a) for a in b[0]],[int(a) for a in b[1]])
            temp_sent.append(torch.LongTensor(b[0]))
            while(len(b[1])<max_box):
                b[1].append(0)
            temp_box.append(torch.LongTensor(b[1]))
        train_box.append(torch.stack(temp_box, dim=0))
        train_sent.append(torch.stack(temp_sent, dim=0))
    return train_box, train_sent
