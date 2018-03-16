import torch
import numpy as np


def batchify(data, batchsize, verbose):
    # data_store = [[corpus.train_value, corpus.train_field , corpus.train_ppos, corpus.train_pneg, corpus.train_sent],\
    # [corpus.test_value, corpus.test_field , corpus.test_ppos, corpus.test_pneg, corpus.test_sent],\
    # [corpus.valid_value, corpus.valid_field , corpus.valid_ppos, corpus.valid_pneg, corpus.valid_sent]]
    #
    # data_dict = [[corpus.train_value_dict, corpus.train_field_dict , corpus.train_ppos_dict, corpus.train_pneg_dict, corpus.train_sent_dict],\
    # [corpus.test_value_dict, corpus.test_field_dict , corpus.test_ppos_dict, corpus.test_pneg_dict, corpus.test_sent_dict],\
    # [corpus.valid_value_dict, corpus.valid_field_dict , corpus.valid_ppos_dict, corpus.valid_pneg_dict, corpus.valid_sent_dict]]

    sent = []
    sent_length = []
    value = []
    value_length = []
    field = []
    field_length = []
    ppos = []
    ppos_length = []
    pneg = []
    pneg_length = []

    data_padded = [value, field, ppos, pneg, sent]

    data_length = [value_length, field_length, ppos_length, pneg_length, sent_length]

    datum = sorted(zip(data[4], data[0], data[1], data[2], data[3]), key=lambda tup: len(tup[0]))
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
        data_length[4].append(temp_sentences_actual_length)
        data_padded[4].append(torch.stack(temp_sentences_padded, dim=0))
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
        data_padded[3].append(torch.stack(temp_table_pneg_padded, dim=0))
        data_padded[2].append(torch.stack(temp_table_ppos_padded, dim=0))
        data_padded[1].append(torch.stack(temp_table_field_padded, dim=0))
        data_padded[0].append(torch.stack(temp_table_value_padded, dim=0))
        data_length[3].append(temp_table_pneg_actual_length)
        data_length[2].append(temp_table_ppos_actual_length)
        data_length[1].append(temp_table_field_actual_length)
        data_length[0].append(temp_table_value_actual_length)
    return data_padded, data_length
