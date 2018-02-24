import torchwordemb
import argparse
import torch
import data as data
from batchify_pad import batchify, pad_collate
###############################################################################
# Parse command line arguments
###############################################################################

parser = argparse.ArgumentParser(description='PyTorch Text Generation Model')
parser.add_argument('--cuda', action='store_true', default=True, help='use CUDA')
parser.add_argument('--seed', type=int, default=1,help='random seed')
parser.add_argument('--batchsize', type=int, default=32,help='batchsize')
parser.add_argument('--data', type=str, default='./data/Wiki-Data/wikipedia-biography-dataset/',help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=200,help='size of word embeddings')
parser.add_argument('--nlayers', type=int, default=2,help='number of layers')
parser.add_argument('--nhid', type=int, default=200,help='number of hidden units per layer')
parser.add_argument('--dropout', type=float, default=0.2,help='dropout applied to layers (0 = no dropout)')
args = parser.parse_args()
cuda = args.cuda
dropout = args.dropout
seed = args.seed
num_layers = args.nlayers
emb_size = args.emsize
hidden_size = args.nhid
batchsize = args.batchsize
data_path = args.data
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
corpus = data.Corpus(data_path)
#Use: table: corpus.train[i].box, bio: corpus.train[i].sent
#Sentence are \n seprated and box is \t seprated
print('Train:', len(corpus.train), 'Validation:', len(corpus.val), 'Test:', len(corpus.test))
###############################################################################
# Batchify, Padding, Stacking
###############################################################################
train_box, train_sent, val_box, val_sent, test_box, test_sent = batchify(corpus)
datm_list = [(train_box, train_sent), (val_box, val_sent), (test_box, test_sent)]
for datum in datm_list:
    train_box, train_sent = pad_collate(datum, batchsize)
###############################################################################
# TODO: Build Model
# Train Model
# Test Model
###############################################################################
"""vocab_size = len(corpus.dictionary)
model = model.RNNModel(dropout=dropout, vocab_size=vocab_size, input_dim=emb_size, num_layers=num_layers, hidden_size=hidden_size)
if args.cuda:
    model.cuda()"""
