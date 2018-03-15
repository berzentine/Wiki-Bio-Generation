import torch
import numpy as np


def batchify(corpus, batchsize, verbose):
    data_store = [[corpus.train_value, corpus.train_field , corpus.train_ppos, corpus.train_pneg, corpus.train_sent],\
    [corpus.test_value, corpus.test_field , corpus.test_ppos, corpus.test_pneg, corpus.test_sent],\
    [corpus.valid_value, corpus.valid_field , corpus.valid_ppos, corpus.valid_pneg, corpus.valid_sent]]

    data_dict = [[corpus.train_value_dict, corpus.train_field_dict , corpus.train_ppos_dict, corpus.train_pneg_dict, corpus.train_sent_dict],\
    [corpus.test_value_dict, corpus.test_field_dict , corpus.test_ppos_dict, corpus.test_pneg_dict, corpus.test_sent_dict],\
    [corpus.valid_value_dict, corpus.valid_field_dict , corpus.valid_ppos_dict, corpus.valid_pneg_dict, corpus.valid_sent_dict]]

    train_sent = []
    train_sent_length = []
    train_value = []
    train_value_length = []
    train_field = []
    train_field_length = []
    train_ppos = []
    train_ppos_length = []
    train_pneg = []
    train_pneg_length = []

    test_sent = []
    test_sent_length = []
    test_value = []
    test_value_length = []
    test_field = []
    test_field_length = []
    test_ppos = []
    test_ppos_length = []
    test_pneg = []
    test_pneg_length = []

    valid_sent = []
    valid_sent_length = []
    valid_value = []
    valid_value_length = []
    valid_field = []
    valid_field_length = []
    valid_ppos = []
    valid_ppos_length = []
    valid_pneg = []
    valid_pneg_length = []


    data_padded = [[train_value,train_field,train_ppos,train_pneg, train_sent],\
    [test_value,test_field,test_ppos,test_pneg, test_sent],\
    [valid_value,valid_field,valid_ppos,valid_pneg, valid_sent]]

    data_length = [[train_value_length, train_field_length, train_ppos_length, train_pneg_length, train_sent_length],\
    [test_value_length, test_field_length, test_ppos_length, test_pneg_length, test_sent_length],\
    [valid_value_length, valid_field_length, valid_ppos_length, valid_pneg_length, valid_sent_length]]



    for i in range(0,3):
        if verbose:
            if i ==1: print('Done batchifying train data')
            if i ==2: print('Done batchifying test data')
        datum = sorted(zip(data_store[i][4], data_store[i][0], data_store[i][1], data_store[i][2], data_store[i][3]), key=lambda tup: len(tup[0]))
        for d in range(0,len(datum), batchsize):
            temp_sentences_actual_length = []
            temp_sentences_padded = []
            temp_table_field_actual_length = []
            temp_table_field_padded = []
            temp_table_value_actual_length = []
            temp_table_value_padded = []
            temp_table_ppos_actual_length = []
            temp_table_ppos_padded = []
            temp_table_pneg_actual_length = []
            temp_table_pneg_padded = []
            # padd sentences in batch
            batch_sent = datum[d:d+batchsize]
            max_sent = len(batch_sent[-1][0])
            for b in batch_sent:
                temp_sentences_actual_length.append(len(b[0]))
                while(len(b[0])<max_sent):
                    b[0].append(0)
                temp_sentences_padded.append(torch.LongTensor(b[0]))
            data_length[i][4].append(temp_sentences_actual_length)
            data_padded[i][4].append(torch.stack(temp_sentences_padded, dim=0))
            # padd remaining items in the batch
            # padd values in batch # padd fields in batch # padd positions in batch
            # can be done toegther since same length of these all
            table_max_length = -100
            for b in batch_sent: # find biggest table
                if table_max_length<len(b[1]):
                    table_max_length=len(b[1])
            for b in batch_sent:
                temp_table_value_actual_length.append(len(b[1]))
                temp_table_field_actual_length.append(len(b[2]))
                temp_table_ppos_actual_length.append(len(b[3]))
                temp_table_pneg_actual_length.append(len(b[4]))
                while(len(b[1])<table_max_length): # done based on value
                    b[1].append(0) # value
                    b[2].append(0) # field
                    b[3].append(0) # ppos
                    b[4].append(0) # ppneg
                temp_table_pneg_padded.append(torch.LongTensor(b[4]))
                temp_table_ppos_padded.append(torch.LongTensor(b[3]))
                temp_table_field_padded.append(torch.LongTensor(b[2]))
                temp_table_value_padded.append(torch.LongTensor(b[1]))
            # append padded sentences and their lengths in final batch
            data_padded[i][3].append(torch.stack(temp_table_pneg_padded, dim=0))
            data_padded[i][2].append(torch.stack(temp_table_ppos_padded, dim=0))
            data_padded[i][1].append(torch.stack(temp_table_field_padded, dim=0))
            data_padded[i][0].append(torch.stack(temp_table_value_padded, dim=0))
            data_length[i][3].append(temp_table_pneg_actual_length)
            data_length[i][2].append(temp_table_ppos_actual_length)
            data_length[i][1].append(temp_table_field_actual_length)
            data_length[i][0].append(temp_table_value_actual_length)
        #break
        if verbose: print('Done batchifying valid data')
    return data_padded, data_length
