# should take test data as input
# spit out the generated text and the original text
import torchwordemb
import math
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import data_reader as data_reader
import random
from batchify_pad import batchify
from models.joint_model import Seq2SeqModel
from utils.plot_utils import plot
from models.ConditionedLM import ConditionedLM
from torch.autograd import Variable

###############################################################################
# Parse command line arguments
###############################################################################

parser = argparse.ArgumentParser(description='PyTorch Text Generation Model')
parser.add_argument('--cuda', action='store_true', default=False, help='use CUDA')
parser.add_argument('--verbose', action='store_true', default=False, help='use Verbose')
parser.add_argument('--limit', type=float, default=0.05,help='limit size of data')
parser.add_argument('--seed', type=int, default=1,help='random seed')
parser.add_argument('--batchsize', type=int, default=32,help='batchsize')
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
corpus = data_reader.Corpus(data_path, vocab_path, 1, limit, verbose)
WORD_VOCAB_SIZE = len(corpus.word_vocab)
FIELD_VOCAB_SIZE = len(corpus.field_vocab)
POS_VOCAB_SIZE = len(corpus.pos_vocab)
print('='*32)

corpus.train_value, corpus.train_value_len, corpus.train_field, corpus.train_field_len, corpus.train_ppos, corpus.train_ppos_len, \
corpus.train_pneg, corpus.train_pneg_len, corpus.train_sent, corpus.train_sent_len = batchify([corpus.train_value, corpus.train_field , corpus.train_ppos, corpus.train_pneg, corpus.train_sent], batchsize, verbose)

corpus.test_value, corpus.test_value_len, corpus.test_field, corpus.test_field_len, corpus.test_ppos, corpus.test_ppos_len, \
corpus.test_pneg, corpus.test_pneg_len, corpus.test_sent, corpus.test_sent_len = batchify([corpus.test_value, corpus.test_field , corpus.test_ppos, corpus.test_pneg, corpus.test_sent], 1, verbose)

corpus.valid_value, corpus.valid_value_len, corpus.valid_field, corpus.valid_field_len, corpus.valid_ppos, corpus.valid_ppos_len, \
corpus.valid_pneg, corpus.valid_pneg_len, corpus.valid_sent, corpus.valid_sent_len = batchify([corpus.valid_value, corpus.valid_field , corpus.valid_ppos, corpus.valid_pneg, corpus.valid_sent], batchsize, verbose)

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
    value = data_source['value'][num]
    ppos = data_source['ppos'][num]
    pneg = data_source['pneg'][num]
    sent = batch[:, 0:batch.size(1)-1]
    actual_sent =  batch[:, 0:batch.size(1)]
    target = batch[:, 1:batch.size(1)]
    sent_len = data_source['sent_len'][num]
    # data = torch.stack(data)
    # target = torch.stack(target)
    if cuda:
        sent = sent.cuda()
        target = target.cuda()
        field = field.cuda()
        value = value.cuda()
        ppos = ppos.cuda()
        pneg = pneg.cuda()
    sent = Variable(sent, volatile=evaluation)
    field = Variable(field, volatile=evaluation)
    value = Variable(value, volatile=evaluation)
    ppos = Variable(ppos, volatile=evaluation)
    pneg = Variable(pneg, volatile=evaluation)
    target = Variable(target)
    return sent, sent_len, ppos, pneg, field, value, target, actual_sent


test_batches = [x for x in range(0, len(corpus.test["sent"]))]

def test_evaluate(data_source, data_order, test):
    with open('docID.txt', 'w') as wp:
        #index=0
        with open('reference.txt', 'w') as rp:
            with open('generated.txt', 'w') as gp:
                total_loss = total_words = 0
                model.eval()
                start_time = time.time()
                random.shuffle(data_order)
                for batch_num in data_order:
                    sent, sent_len, ppos, pneg, field, value, target, actual_sent = get_data(data_source, batch_num, True)
                    ref_seq = []
                    for i in range(0, len(actual_sent[0])):
                        ref_seq.append(corpus.word_vocab.idx2word[int(actual_sent[0][i])])
                        #if WORD_VOCAB_SIZE>int(sent[0][i]):
                        #    ref_seq.append(corpus.word_vocab.idx2word[int(sent[0][i])])
                    gen_seq = model.generate(value, field, ppos, pneg, 1, False, max_length, \
                                                           corpus.word_vocab.word2idx["<sos>"],  corpus.word_vocab.word2idx["<eos>"], corpus.word_vocab)
                    #print ref_seq
                    #print gen_seq
                    #index+=1
                    #print '='*32
                    for r in ref_seq:
                        rp.write(r+" ")
                    for g in gen_seq:
                        gp.write(g+" ")
                    #wp.write("DOCID: "+str(index))
                    rp.write("\n\n")
                    gp.write("\n\n")
                    #wp.write("\n\n")
    import os
    os.system("echo \"************ Non-tokenized scores ************\"")
    os.system("./Scoring_scripts/multi-bleu.pl reference.txt < generated.txt | grep \"BLEU\"")
    os.system("echo \"======================================================\"")
    os.system("echo \"************ Tokenized scores ************\"")
    os.system("./Scoring_scripts/tokenizer.pl -l en < reference.txt > tokenized_reference.txt")
    os.system("./Scoring_scripts/tokenizer.pl -l en < generated.txt > tokenized_generated.txt")
    os.system("./Scoring_scripts/multi-bleu.pl tokenized_reference.txt < tokenized_generated.txt | grep \"BLEU\"")
    #
    return

# Load the best saved model.
with open(model_save_path+"best_model.pth", 'rb') as f:
    model = torch.load(f)
# Run on test data.
test_evaluate(corpus.test, test_batches, test=True)
