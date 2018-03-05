import torchwordemb
import math
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import data_handler as data_handler
from batchify_pad import batchify, pad_collate
from utils.plot_utils import plot
from models.ConditionedLM import ConditionedLM
from torch.autograd import Variable

###############################################################################
# Parse command line arguments
###############################################################################

parser = argparse.ArgumentParser(description='PyTorch Text Generation Model')
parser.add_argument('--cuda', action='store_true', default=False, help='use CUDA')
parser.add_argument('--seed', type=int, default=1,help='random seed')
parser.add_argument('--batchsize', type=int, default=32,help='batchsize')
parser.add_argument('--lr', type=int, default=0.1,help='learning rate')
parser.add_argument('--data', type=str, default='./data/Wiki-Data/wikipedia-biography-dataset/',help='location of the data corpus')
parser.add_argument('--model_save_path', type=str, default='./saved_models/best_model.pth',help='location of the best model to save')
parser.add_argument('--plot_save_path', type=str, default='./saved_models/loss_plot.png',help='location of the loss plot to save')
parser.add_argument('--emsize', type=int, default=200,help='size of word embeddings')
parser.add_argument('--nlayers', type=int, default=1,help='number of layers')
parser.add_argument('--nhid', type=int, default=100,help='number of hidden units per layer')
parser.add_argument('--dropout', type=float, default=0.2,help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--clip', type=float, default=0.2,help='gradient clip')
parser.add_argument('--log_interval', type=float, default=500,help='log interval')
parser.add_argument('--epochs', type=int, default=50,help='epochs')

args = parser.parse_args()
cuda = args.cuda
total_epochs = args.epochs
dropout = args.dropout
seed = args.seed
num_layers = args.nlayers
emb_size = args.emsize
hidden_size = args.nhid
batchsize = args.batchsize
data_path = args.data
plot_save_path = args.plot_save_path
model_save_path = args.model_save_path
lr = args.lr
clip = args.clip
log_interval = args.log_interval
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
corpus = data_handler.Corpus(data_path)
#Use: table: corpus.train[i].box, bio: corpus.train[i].sent
#Sentence are \n seprated and box is \t seprated
print('Train:', len(corpus.train), 'Validation:', len(corpus.val), 'Test:', len(corpus.test))
###############################################################################
# Batchify, Padding, Stacking
###############################################################################
train_box, train_sent, val_box, val_sent, test_box, test_sent = batchify(corpus)
# datm_list = [(train_box, train_sent), (val_box, val_sent), (test_box, test_sent)]
# padded_data = []
# for datum in datm_list:
#     train_box, train_sent, train_box_length, train_sent_length = pad_collate(datum, batchsize)
#     padded_data.append((train_box, train_box_length, train_sent, train_sent_length))

train_box, train_sent, train_box_length, train_sent_length = pad_collate((train_box, train_sent), batchsize)
val_box, val_sent, val_box_length, val_sent_length = pad_collate((val_box, val_sent), batchsize)
test_box, test_sent, test_box_length, test_sent_length = pad_collate((test_box, test_sent), batchsize)
