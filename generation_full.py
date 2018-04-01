# should take test data as input
# spit out the generated text and the original text
import torchwordemb
import math
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import data_reader_replicated as data_reader
import random
from batchify_pad import batchify
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from models.joint_model import Seq2SeqModel
from utils.plot_utils import plot
from torch.autograd import Variable

###############################################################################
# Parse command line arguments
###############################################################################

parser = argparse.ArgumentParser(description='PyTorch Text Generation Model')
parser.add_argument('--cuda', action='store_true', default=False, help='use CUDA')
parser.add_argument('--verbose', action='store_true', default=False, help='use Verbose')
parser.add_argument('--limit', type=float, default=0.05,help='limit size of data')
parser.add_argument('--seed', type=int, default=1,help='random seed')
parser.add_argument('--batchsize', type=int, default=1,help='batchsize')
parser.add_argument('--lr', type=int, default=0.0005,help='learning rate')
parser.add_argument('--data', type=str, default='./data/Wiki-Data/wikipedia-biography-dataset/',help='location of the data corpus')
parser.add_argument('--vocab', type=str, default='./data/Wiki-Data/vocab/', help='location of the vocab files')
parser.add_argument('--model_save_path', type=str, default='./saved_models/best_model.pth',help='location of the best model to save')
parser.add_argument('--plot_save_path', type=str, default='./saved_models/loss_plot.png',help='location of the loss plot to save')
parser.add_argument('--word_emsize', type=int, default=400,help='size of word embeddings')
parser.add_argument('--field_emsize', type=int, default=50,help='size of field embeddings')
parser.add_argument('--pos_emsize', type=int, default=5,help='size of position embeddings')
parser.add_argument('--nlayers', type=int, default=1,help='number of layers')
parser.add_argument('--nhid', type=int, default=500,help='number of hidden units per layer')
parser.add_argument('--dropout', type=float, default=0.2,help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--clip', type=float, default=0.2,help='gradient clip')
parser.add_argument('--log_interval', type=float, default=500,help='log interval')
parser.add_argument('--epochs', type=int, default=100,help='epochs')
parser.add_argument('--max_sent_length', type=int, default=40,help='maximum sentence length for decoding')
parser.add_argument('--ref_path', type=str, required=True, help='Path for storing the reference file')
parser.add_argument('--gen_path', type=str, required=True, help='Path for storing the generated file')
parser.add_argument('--unk_gen_path', type=str, required=True, help='Path for storing the unk replaced generated file')

"""
USAGE: python generation.py --limit=0.001 --ref_path=reference.txt --gen_path=generated.txt
Outputs:
reference.txt : Gold text for comparision will be stored here
generated.txt : System generated text will be stored here
reference.txt.tokenized : Tokenized version of reference.txt will be stored here
generated.txt.tokenized : Tokenized version of generated.txt will be stored here
"""
args = parser.parse_args()
cuda = args.cuda
verbose = args.verbose
limit = args.limit
total_epochs = args.epochs
dropout = args.dropout
seed = args.seed
num_layers = args.nlayers
word_emb_size = args.word_emsize
field_emb_size = args.field_emsize
pos_emb_size = args.pos_emsize
hidden_size = args.nhid
batchsize = args.batchsize
data_path = args.data
vocab_path = args.vocab
plot_save_path = args.plot_save_path
model_save_path = args.model_save_path
lr = args.lr
clip = args.clip
log_interval = args.log_interval
max_length = args.max_sent_length
ref_path = args.ref_path
gen_path = args.gen_path
unk_gen_path = args.unk_gen_path


print("Load embedding")
emb_path = "../word2vec/GoogleNews-vectors-negative300.bin"
#w2v_vocab, emb_vec = torchwordemb.load_word2vec_bin(emb_path)

torch.manual_seed(seed)
if torch.cuda.is_available():
    if not cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(seed)

print("Load data")
corpus = data_reader.Corpus(data_path, vocab_path, batchsize, limit, verbose)
WORD_VOCAB_SIZE = len(corpus.word_vocab)
FIELD_VOCAB_SIZE = len(corpus.field_vocab)
POS_VOCAB_SIZE = len(corpus.pos_vocab)
print('='*32)


corpus.train_value, corpus.train_value_len, corpus.train_field, corpus.train_field_len, corpus.train_ppos, corpus.train_ppos_len, \
corpus.train_pneg, corpus.train_pneg_len, corpus.train_sent, corpus.train_sent_len , corpus.train_ununk_sent, \
corpus.train_ununk_field, corpus.train_ununk_value, corpus.train_sent_mask, corpus.train_value_mask  = \
batchify([corpus.train_value, corpus.train_field , corpus.train_ppos, corpus.train_pneg, corpus.train_sent], \
batchsize, verbose, [corpus.train_ununk_sent, corpus.train_ununk_field, corpus.train_ununk_value])

corpus.test_value, corpus.test_value_len, corpus.test_field, corpus.test_field_len, corpus.test_ppos, corpus.test_ppos_len, \
corpus.test_pneg, corpus.test_pneg_len, corpus.test_sent, corpus.test_sent_len, \
corpus.test_ununk_sent, corpus.test_ununk_field, corpus.test_ununk_value, corpus.test_sent_mask, corpus.test_value_mask  = \
batchify([corpus.test_value, corpus.test_field , corpus.test_ppos, corpus.test_pneg, corpus.test_sent], \
batchsize, verbose, [corpus.test_ununk_sent, corpus.test_ununk_field, corpus.test_ununk_value])

corpus.valid_value, corpus.valid_value_len, corpus.valid_field, corpus.valid_field_len, corpus.valid_ppos, corpus.valid_ppos_len, \
corpus.valid_pneg, corpus.valid_pneg_len, corpus.valid_sent, corpus.valid_sent_len,\
corpus.valid_ununk_sent, corpus.valid_ununk_field, corpus.valid_ununk_value, corpus.valid_sent_mask, corpus.valid_value_mask = \
batchify([corpus.valid_value, corpus.valid_field , corpus.valid_ppos, corpus.valid_pneg, corpus.valid_sent], \
batchsize, verbose, [corpus.valid_ununk_sent, corpus.valid_ununk_field, corpus.valid_ununk_value] )


corpus.create_data_dictionaries()

if verbose:
    print('='*15, 'SANITY CHECK', '='*15)
    print('='*3, '# P +', '='*3, '# P -', '='*3, '# F', '='*3, '# V(F)', '='*3, '# Sent', '-- Should be equal across rows --')
    print(len(corpus.test_ppos), len(corpus.test_pneg), len(corpus.test_field), len(corpus.test_value), len(corpus.test_sent))

    print('='*3, '# PLen +', '='*3, '# PLen -', '='*3, '# FLen', '='*3, '# V(F)Len', '='*3, '# SentLen', '-- Should be equal across rows --')
    print(len(corpus.test_ppos_len), len(corpus.test_pneg_len), len(corpus.test_field_len), len(corpus.test_value_len), len(corpus.test_sent_len))
    print('='*32)


def get_data(data_source, num, evaluation):
    batch = data_source['sent'][num]
    field = data_source['field'][num]

    field_ununk = data_source['field_ununk'][num]
    value_ununk = data_source['value_ununk'][num]
    sent_ununk = data_source['sent_ununk'][num]

    value = data_source['value'][num]
    ppos = data_source['ppos'][num]
    pneg = data_source['pneg'][num]
    sent = batch[:, 0:batch.size(1)-1]
    actual_sent =  batch[:, 0:batch.size(1)]
    target = batch[:, 1:batch.size(1)]
    sent_len = data_source['sent_len'][num]
    value_len = data_source['value_len'][num]
    sent_mask = data_source['sent_mask'][num]
    value_mask = data_source['value_mask'][num]
    # data = torch.stack(data)
    # target = torch.stack(target)
    if cuda:
        sent = sent.cuda()
        target = target.cuda()
        field = field.cuda()
        value = value.cuda()
        ppos = ppos.cuda()
        pneg = pneg.cuda()
        value_mask = value_mask.cuda()
    sent = Variable(sent, volatile=evaluation)
    field = Variable(field, volatile=evaluation)
    value = Variable(value, volatile=evaluation)
    ppos = Variable(ppos, volatile=evaluation)
    pneg = Variable(pneg, volatile=evaluation)
    value_mask = Variable(value_mask, volatile=evaluation)

    field_ununk = Variable(field_ununk, volatile=evaluation)
    value_ununk = Variable(value_ununk, volatile=evaluation)
    sent_ununk = Variable(sent_ununk, volatile=evaluation)

    target = Variable(target)
    return sent, sent_len, ppos, pneg, field, value, value_len, target, actual_sent, sent_ununk, field_ununk , value_ununk, sent_mask, value_mask


test_batches = [x for x in range(0, len(corpus.test["sent"]))]
train_batches = [x for x in range(0, len(corpus.train["sent"]))]

def generate(value, value_len, field, ppos, pneg, batch_size, \
            train, max_length, start_symbol, end_symbol, dictionary, unk_symbol, \
            ununk_dictionary, value_ununk, value_mask):
    input_d = model.sent_lookup(value)
    input_z = torch.cat((model.field_lookup(field), model.ppos_lookup(ppos), model.pneg_lookup(pneg)), 2)
    encoder_initial_hidden = model.encoder.init_hidden(batch_size, model.encoder_hidden_size)
    if cuda:
        encoder_initial_hidden = encoder_initial_hidden.cuda()
    encoder_output, encoder_hidden = model.encoder.forward(input_d=input_d, input_z=input_z, hidden=encoder_initial_hidden, mask=value_mask)
    encoder_output = torch.stack(encoder_output, dim=0)
    encoder_hidden = (encoder_hidden[0].unsqueeze(0), encoder_hidden[1].unsqueeze(0))
    gen_seq = []
    unk_rep_seq = []
    start_symbol =  Variable(torch.LongTensor(1,1).fill_(start_symbol))
    if cuda:
        start_symbol = start_symbol.cuda()
    curr_input = model.sent_lookup(start_symbol) # TODO: change here to look and handle batches
    # print curr_input.shape()
    prev_hidden = encoder_hidden
    for i in range(max_length):
        decoder_output, prev_hidden, attn_vector = model.decoder.forward_biased_lstm(input=curr_input, hidden=prev_hidden, encoder_hidden=encoder_output, input_z=input_z, mask=value_mask)
        max_val, max_idx = torch.max(decoder_output.squeeze(), 0)
        curr_input = model.sent_lookup(max_idx).unsqueeze(0)
        # TODO: Issue here
        # print curr_input.shape()
        # exit(0)
        if dictionary.idx2word[int(max_idx)] == '<eos>':
            break
        if int(max_idx) == unk_symbol:
            if cuda:
                value_ununk = value_ununk.cuda()
            unk_max_val, unk_max_idx = torch.max(attn_vector[0][0,:value_len[0],0], 0)
            sub = value_ununk[0][unk_max_idx] # should be value_ununk
            word = ununk_dictionary.idx2word[int(sub)] # should be replaced from ununk dictionary word_ununk_vocab
            print("Unk got replaced with", word)
        else:
            word = dictionary.idx2word[int(max_idx)]
        gen_seq.append(dictionary.idx2word[int(max_idx)])
        unk_rep_seq.append(word)
        if dictionary.idx2word[int(max_idx)] == '<eos>':
            break
    return gen_seq, unk_rep_seq



def test_evaluate(data_source, data_order, test):
    gold_set = []
    pred_set = []
    unk_set = []
    with open(unk_gen_path, 'w') as up:
        with open(ref_path, 'w') as rp:
            with open(gen_path, 'w') as gp:
                total_loss = total_words = 0
                model.eval()
                start_time = time.time()
                random.shuffle(data_order)
                for batch_num in data_order:
                    sent, sent_len, ppos, pneg, field, value, value_len, target, actual_sent, sent_ununk, field_ununk ,\
                    value_ununk, sent_mask, value_mask = get_data(data_source, batch_num, True)
                    ref_seq = []
                    for i in range(1, len(actual_sent[0])):
                        ref_seq.append(corpus.word_ununk_vocab.idx2word[int(sent_ununk[0][i])]) # changed here
                        #if WORD_VOCAB_SIZE>int(sent[0][i]):
                        #    ref_seq.append(corpus.word_vocab.idx2word[int(sent[0][i])])
                    gen_seq, unk_rep_seq = generate(value, value_len, field, ppos, pneg, batchsize, False, max_length, \
                                                           corpus.word_vocab.word2idx["<sos>"],  corpus.word_vocab.word2idx["<eos>"], corpus.word_vocab, \
                                                           corpus.word_vocab.word2idx["UNK"], corpus.word_ununk_vocab, value_ununk, value_mask)

                    for u in unk_rep_seq:
                        up.write(u+" ")
                    for r in ref_seq:
                        rp.write(r+" ")
                    for g in gen_seq:
                        gp.write(g+" ")
                    #wp.write("DOCID: "+str(index))
                    up.write("\n\n")
                    rp.write("\n\n")
                    gp.write("\n\n")
                    #wp.write("\n\n")
                    gold_set.append(ref_seq)
                    pred_set.append(gen_seq)
                    unk_set.append(unk_rep_seq)
    import os
    os.system("echo \"************ Python scores ************\"")
    bleu = corpus_bleu(gold_set, pred_set)
    print("WITH UNK Bleu:"+ str(bleu))
    bleu = corpus_bleu(gold_set, unk_set)
    print("WITHOUT UNK Bleu:"+ str(bleu))
    os.system("echo \"************ Non-tokenized scores ************\"")
    os.system("./Scoring_scripts/multi-bleu.pl " +ref_path +" < " +gen_path +" | grep \"BLEU\"")
    os.system("./Scoring_scripts/multi-bleu.pl " +ref_path +" < " +unk_gen_path +" | grep \"BLEU\"")
    os.system("echo \"======================================================\"")
    os.system("echo \"************ Tokenized scores ************\"")
    os.system("./Scoring_scripts/tokenizer.pl -l en < " +ref_path +" > "+ ref_path+".tokenized")
    os.system("./Scoring_scripts/tokenizer.pl -l en < " +gen_path +" > "+ gen_path+".tokenized")
    os.system("./Scoring_scripts/tokenizer.pl -l en < " +unk_gen_path +" > "+ unk_gen_path+".tokenized")
    os.system("./Scoring_scripts/multi-bleu.pl "+ ref_path+".tokenized < "+gen_path+".tokenized | grep \"BLEU\"")
    os.system("./Scoring_scripts/multi-bleu.pl "+ ref_path+".tokenized < "+unk_gen_path+".tokenized | grep \"BLEU\"")
    return

# Load the best saved model.
with open(model_save_path+"best_model.pth", 'rb') as f:
    model = torch.load(f)
# Run on test data.
test_evaluate(corpus.test, test_batches, test=True)
