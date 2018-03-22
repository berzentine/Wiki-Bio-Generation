import torch
import numpy as np


def batchify(data, batchsize, verbose):
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

    datum = sorted(zip(data[4], data[0], data[1], data[2], data[3]), key=lambda tup: len(tup[0]))
    total_batches = len(datum)//batchsize
    for d in range(0,total_batches):
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
        sent_length.append(temp_sentences_actual_length)
        sent.append(torch.stack(temp_sentences_padded, dim=0))
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
        pneg.append(torch.stack(temp_table_pneg_padded, dim=0))
        ppos.append(torch.stack(temp_table_ppos_padded, dim=0))
        field.append(torch.stack(temp_table_field_padded, dim=0))
        value.append(torch.stack(temp_table_value_padded, dim=0))
        pneg_length.append(temp_table_pneg_actual_length)
        ppos_length.append(temp_table_ppos_actual_length)
        field_length.append(temp_table_field_actual_length)
        value_length.append(temp_table_value_actual_length)
    return value, value_length, field, field_length, ppos, ppos_length, pneg, pneg_length, sent, sent_length
