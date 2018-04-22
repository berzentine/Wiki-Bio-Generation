import torchwordemb
import math
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import data_reader_alignments as data_reader
import random
from batchify_pad import batchify, get_batch_alignments
from models.new_dual_model import Seq2SeqDualModel
from utils.plot_utils import plot
from torch.autograd import Variable

###############################################################################
# Parse command line arguments
###############################################################################

parser = argparse.ArgumentParser(description='PyTorch Text Generation Model')
parser.add_argument('--cuda', action='store_true', default=False, help='use CUDA')
parser.add_argument('--verbose', action='store_true', default=False, help='use Verbose')
parser.add_argument('--limit', type=float, default=1,help='limit size of data')
parser.add_argument('--seed', type=int, default=1,help='random seed')
parser.add_argument('--batchsize', type=int, default=32,help='batchsize')
parser.add_argument('--lr', type=float, default=0.0005,help='learning rate')
parser.add_argument('--data', type=str, default='./data/Wiki-Data/wikipedia-biography-dataset/',help='location of the data corpus')
parser.add_argument('--vocab', type=str, default='./data/Wiki-Data/vocab/', help='location of the vocab files')
parser.add_argument('--alignments', type=str, default='./data/Wiki-Data/alignments/', help='location of the alignment files')
parser.add_argument('--use_pickle', action='store_true', default=False, help='Flag whether to use pickled version of alignements or not')
parser.add_argument('--model_save_path', type=str, default='./saved_models/best_model.pth',help='location of the best model to save')
parser.add_argument('--plot_save_path', type=str, default='./saved_models/loss_plot.png',help='location of the loss plot to save')
parser.add_argument('--word_emsize', type=int, default=400,help='size of word embeddings')
parser.add_argument('--field_emsize', type=int, default=50,help='size of field embeddings')
parser.add_argument('--pos_emsize', type=int, default=5,help='size of position embeddings')
parser.add_argument('--nlayers', type=int, default=1,help='number of layers')
parser.add_argument('--nhid', type=int, default=500,help='number of hidden units per layer')
parser.add_argument('--dropout', type=float, default=0.2,help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--clip', type=float, default=5,help='gradient clip')
parser.add_argument('--log_interval', type=float, default=500,help='log interval')
parser.add_argument('--epochs', type=int, default=50,help='epochs')
parser.add_argument('--max_sent_length', type=int, default=64,help='maximum sentence length for decoding')

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
alignment_path = args.alignments
use_pickle = args.use_pickle
plot_save_path = args.plot_save_path
model_save_path = args.model_save_path
lr = args.lr
clip = args.clip
log_interval = args.log_interval
max_length = args.max_sent_length


###############################################################################
# Code Setup
# TODO: Load and make embeddings vector
# Check and Set Cuda
# Load Data and Tokenize it
###############################################################################

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
corpus = data_reader.Corpus(data_path, vocab_path, alignment_path, use_pickle, 1, limit, verbose)
WORD_VOCAB_SIZE = len(corpus.word_vocab)
FIELD_VOCAB_SIZE = len(corpus.field_vocab)
POS_VOCAB_SIZE = len(corpus.pos_vocab)
print('='*32)

corpus.train_value, corpus.train_value_len, corpus.train_field, corpus.train_field_len, corpus.train_ppos, corpus.train_ppos_len, \
corpus.train_pneg, corpus.train_pneg_len, corpus.train_sent, corpus.train_sent_len , corpus.train_ununk_sent, \
corpus.train_ununk_field, corpus.train_ununk_value, corpus.train_sent_mask, corpus.train_value_mask, corpus.train_alignments = \
    batchify([corpus.train_value, corpus.train_field , corpus.train_ppos, corpus.train_pneg, corpus.train_sent], \
             batchsize, verbose, [corpus.train_ununk_sent, corpus.train_ununk_field, corpus.train_ununk_value], corpus.alignments)

corpus.test_value, corpus.test_value_len, corpus.test_field, corpus.test_field_len, corpus.test_ppos, corpus.test_ppos_len, \
corpus.test_pneg, corpus.test_pneg_len, corpus.test_sent, corpus.test_sent_len, \
corpus.test_ununk_sent, corpus.test_ununk_field, corpus.test_ununk_value, corpus.test_sent_mask, corpus.test_value_mask, corpus.test_alignments = \
    batchify([corpus.test_value, corpus.test_field , corpus.test_ppos, corpus.test_pneg, corpus.test_sent], \
             1, verbose, [corpus.test_ununk_sent, corpus.test_ununk_field, corpus.test_ununk_value], corpus.alignments)

corpus.valid_value, corpus.valid_value_len, corpus.valid_field, corpus.valid_field_len, corpus.valid_ppos, corpus.valid_ppos_len, \
corpus.valid_pneg, corpus.valid_pneg_len, corpus.valid_sent, corpus.valid_sent_len, \
corpus.valid_ununk_sent, corpus.valid_ununk_field, corpus.valid_ununk_value, corpus.valid_sent_mask, corpus.valid_value_mask, corpus.valid_alignments = \
    batchify([corpus.valid_value, corpus.valid_field , corpus.valid_ppos, corpus.valid_pneg, corpus.valid_sent], \
             batchsize, verbose, [corpus.valid_ununk_sent, corpus.valid_ununk_field, corpus.valid_ununk_value], corpus.alignments)

corpus.create_data_dictionaries()

if verbose:
    print('='*15, 'SANITY CHECK', '='*15)
    print('='*3, '# P +', '='*3, '# P -', '='*3, '# F', '='*3, '# V(F)', '='*3, '# Sent...and values', '-- Should be equal across rows --')
    print(len(corpus.train_ppos), len(corpus.train_pneg), len(corpus.train_field), len(corpus.train_value), len(corpus.train_sent), \
          len(corpus.train_ununk_sent), len(corpus.train_ununk_value), len(corpus.train_ununk_field))
    print(len(corpus.valid_ppos), len(corpus.valid_pneg), len(corpus.valid_field), len(corpus.valid_value), len(corpus.valid_sent), \
          len(corpus.valid_ununk_sent), len(corpus.valid_ununk_value), len(corpus.valid_ununk_field))
    print(len(corpus.test_ppos), len(corpus.test_pneg), len(corpus.test_field), len(corpus.test_value), len(corpus.test_sent), \
          len(corpus.test_ununk_sent), len(corpus.test_ununk_value), len(corpus.test_ununk_field))

    print('='*3, '# PLen +', '='*3, '# PLen -', '='*3, '# FLen', '='*3, '# V(F)Len', '='*3, '# SentLen', '-- Should be equal across rows --')
    print(len(corpus.train_ppos_len), len(corpus.train_pneg_len), len(corpus.train_field_len), len(corpus.train_value_len), len(corpus.train_sent_len))
    print(len(corpus.valid_ppos_len), len(corpus.valid_pneg_len), len(corpus.valid_field_len), len(corpus.valid_value_len), len(corpus.valid_sent_len))
    print(len(corpus.test_ppos_len), len(corpus.test_pneg_len), len(corpus.test_field_len), len(corpus.test_value_len), len(corpus.test_sent_len))

    print('='*32)
#Build Model and move to CUDA

x = Variable(torch.zeros(1), requires_grad=True)
model = Seq2SeqDualModel(sent_vocab_size=WORD_VOCAB_SIZE, field_vocab_size=FIELD_VOCAB_SIZE, ppos_vocab_size=POS_VOCAB_SIZE,
                         pneg_vocab_size=POS_VOCAB_SIZE, value_vocab_size=WORD_VOCAB_SIZE, sent_embed_size=word_emb_size,
                         field_embed_size=field_emb_size, value_embed_size=word_emb_size, ppos_embed_size=pos_emb_size,
                         pneg_embed_size=pos_emb_size, encoder_hidden_size=hidden_size, decoder_hidden_size=hidden_size,
                         decoder_num_layer=num_layers, verbose=verbose, cuda_var=cuda,x=x)

weight_mask = torch.ones(WORD_VOCAB_SIZE)
if args.cuda:
    model.cuda()
    weight_mask = torch.ones(WORD_VOCAB_SIZE).cuda()


#Build criterion and optimizer

weight_mask[0] = 0
criterion = nn.CrossEntropyLoss(ignore_index=0, weight=weight_mask, size_average=True)
#criterion1 = nn.CrossEntropyLoss(ignore_index=0, size_average=False)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
#optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

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
    alignments = get_batch_alignments(value, corpus.alignments)
    # alignments = data_source['alignments'][num]
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
        alignments = alignments.cuda()
    sent = Variable(sent, volatile=evaluation)
    field = Variable(field, volatile=evaluation)
    value = Variable(value, volatile=evaluation)
    ppos = Variable(ppos, volatile=evaluation)
    pneg = Variable(pneg, volatile=evaluation)
    value_mask = Variable(value_mask, volatile=evaluation)
    alignments = Variable(alignments)
    field_ununk = Variable(field_ununk, volatile=evaluation)
    value_ununk = Variable(value_ununk, volatile=evaluation)
    sent_ununk = Variable(sent_ununk, volatile=evaluation)

    target = Variable(target)
    return sent, sent_len, ppos, pneg, field, value, value_len, target, actual_sent, sent_ununk,\
           field_ununk , value_ununk, sent_mask, value_mask, alignments


train_batches = [x for x in range(0, len(corpus.train["sent"]))]
val_batches = [x for x in range(0, len(corpus.valid["sent"]))]
test_batches = [x for x in range(0, len(corpus.test["sent"]))]

def train():
    batch_loss = batch_words = i = 0
    total_loss = total_words = 0
    model.train()
    start_time = time.time()
    num_batches = 0
    #random.shuffle(train_batches)
    losses = []
    batch_losses = []
    for batch_num in train_batches:
        i+=1
        num_batches+=1
        #batch_num = 21
        sent, sent_len, ppos, pneg, field, value, value_len, target, actual_sent, sent_ununk, \
        field_ununk , value_ununk, sent_mask, value_mask, alignments  = get_data(corpus.train, batch_num, False)
        #print sent.shape, sent_len
        decoder_output, decoder_hidden = model.forward_with_attn(sent, value, field, ppos, pneg, batchsize, value_mask, alignments)
        loss = 0
        words = 0
        #for bsz in range(decoder_output.size(0)):
        # print(decoder_output[bsz, sent_len[bsz], :].size(), target[bsz, sent_len[bsz]].size())
        #    loss1 += criterion1(decoder_output[bsz, 0:sent_len[bsz], :], target[bsz, 0:sent_len[bsz]])
        #print(decoder_output[bsz, 0:sent_len[bsz], :])
        #print(batch_num)
        #print(target[bsz, 0:sent_len[bsz]])
        #print(criterion(decoder_output[bsz, 0:sent_len[bsz], :], target[bsz, 0:sent_len[bsz]]), loss.data)
        #for di in range(decoder_output.size(1)): # decoder_output = batch_size X seq_len X vocabsize
        #     #best_vocab, best_index = decoder_output[:,di,:].data.topk(1)
        #    loss += criterion(decoder_output[:, di, :].squeeze(), target[:, di])
        #print(decoder_output.size())
        #print(decoder_output.view(-1, WORD_VOCAB_SIZE).size())
        #print(target.size())
        #print(target.view(-1).size())
        #print(loss.data)
        # decoder_output = decoder_output.view(-1, WORD_VOCAB_SIZE).contiguous()
        loss = criterion(decoder_output.view(-1, WORD_VOCAB_SIZE), target.contiguous().view(-1))
        #print(sent_len)
        #print(loss.data)
        #exit(0)
        losses.append(loss.data[0])
        batch_losses.append(loss.data[0])
        #print(losses)
        total_loss += loss.data
        total_words += sum(sent_len)
        #print(total_loss, total_words)
        batch_loss += loss.data
        batch_words += sum(sent_len)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()
        #print batch_loss[0]
        if i % log_interval == 0 and i > 0:
            #cur_loss = batch_loss[0] / batch_words
            #print(cur_loss)
            #print(batch_losses)
            cur_loss = np.mean(batch_losses)
            #print(cur_loss)
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.6f} | ppl {:8.6f}'.format(
                epoch, i, len(train_batches), lr, elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            batch_loss = 0
            batch_words = 0
            num_batches=0
            batch_losses = []
            start_time = time.time()
        #exit(0)

        del sent, sent_len, ppos, pneg, field, value, value_len, target, actual_sent, sent_ununk, \
            field_ununk , value_ununk, sent_mask, value_mask

    #print(total_loss[0], total_words)
    train_losses.append(np.mean(losses))


def test_evaluate(data_source, data_order, test):
    """total_loss = total_words = 0
    model.eval()
    start_time = time.time()
    random.shuffle(data_order)
    for batch_num in data_order:
        sent, sent_len, ppos, pneg, field, value, value_len, target, actual_sent = get_data(data_source, batch_num, True)
        gen_seq, unk_rep_seq = model.generate(value, value_len, field, ppos, pneg, 1, False, max_length, \
                                                corpus.word_vocab.word2idx["<sos>"],  corpus.word_vocab.word2idx["<eos>"], corpus.word_vocab, corpus.word_vocab.word2idx["UNK"])

    """
    return 0 # TODO

def evaluate(data_source, data_order, test):
    total_loss = total_words = 0
    model.eval()
    start_time = time.time()
    random.shuffle(data_order)
    losses = []
    for batch_num in data_order:
        sent, sent_len, ppos, pneg, field, value, value_len, target, actual_sent, sent_ununk, field_ununk , \
        value_ununk, sent_mask, value_mask, alignments = get_data(data_source, batch_num, True)
        decoder_output, decoder_hidden = model.forward_with_attn(sent, value, field, ppos, pneg, batchsize, value_mask, alignments)
        """decoder_output, decoder_hidden = model.generate(value, field, ppos, pneg, batchsize, False, max_length, \
                                                    corpus.word_vocab.word2idx["<sos>"],  corpus.word_vocab.word2idx["<eos>"], corpus.word_vocab)"""

        loss = 0
        loss = criterion(decoder_output.view(-1, WORD_VOCAB_SIZE), target.contiguous().view(-1))
        #for di in range(decoder_output.size(1)): # decoder_output = batch_len X seq_len X vocabsize
        #best_vocab, best_index = decoder_output[:,di,:].data.topk(1)
        #    loss += criterion(decoder_output[:, di, :].squeeze(), target[:,di])
        losses.append(loss.data[0])
        total_loss += loss.data
        total_words += sum(sent_len)
        del sent, sent_len, ppos, pneg, field, value, value_len, target, actual_sent, sent_ununk, \
            field_ununk , value_ununk, sent_mask, value_mask
    return np.mean(losses)

best_val_loss = None
best_train_loss = None
val_losses = []
train_losses = []
try:
    for epoch in range(1, total_epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(corpus.valid, val_batches, test=False)
        val_losses.append(val_loss)
        tloss = train_losses[-1]
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.6f}s | train_loss {:5.6f} | train ppl {:8.6f} | valid loss {:5.6f} | '
              'valid ppl {:8.6f}'.format(epoch, (time.time() - epoch_start_time), tloss, math.exp(tloss),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        #if not best_val_loss or val_loss < best_val_loss:
        if not best_train_loss or tloss < best_train_loss:
            print("Saving best model")
            with open(model_save_path+"best_model.pth", 'wb') as f:
                torch.save(model, f)
            best_train_loss = tloss

            #best_val_loss = val_loss
            #else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            #    lr /= 2
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

plot(train_losses, val_losses, plot_save_path)

# Load the best saved model.
with open(model_save_path+"best_model.pth", 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = test_evaluate(corpus.test, test_batches, test=True)
print('=' * 89)
print('| End of training | test loss DONT TRUST {:5.6f} | test ppl {:8.6f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
