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
verbose = args.verbose
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
corpus = data_reader.Corpus(data_path, 1, verbose)
print('='*32)
if verbose:
    print('='*15, 'SANITY CHECK', '='*15)
    print('='*3, '# P +', '='*3, '# P -', '='*3, '# F', '='*3, '# V(F)', '='*3, '# Sent', '-- Should be equal across rows --')
    print(len(corpus.train_ppos), len(corpus.train_pneg), len(corpus.train_field), len(corpus.train_value), len(corpus.train_sent))
    print(len(corpus.valid_ppos), len(corpus.valid_pneg), len(corpus.valid_field), len(corpus.valid_value), len(corpus.valid_sent))
    print(len(corpus.test_ppos), len(corpus.test_pneg), len(corpus.test_field), len(corpus.test_value), len(corpus.test_sent))
    print('='*32)
data_padded, data_orig_leng = batchify(corpus, batchsize, verbose)

#Build Model and move to CUDA
model = Seq2SeqModel()
if args.cuda:
    model.cuda()

#Build criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)


def get_data(data_source, num):
    #TODO: return cuda form of data
    pass


train_batches = [x for x in range(0, len(data_padded[0][0]))]
val_batches = [x for x in range(0, len(data_padded[2][0]))]
test_batches = [x for x in range(0, len(data_padded[1][0]))]

def train():
    random.shuffle(train_batches)
    for batch_num in train_batches:
        sent, ppos, pneg, field, value = get_data(data_padded[0], batch_num)
        #TODO: model forward, loss calculation and debug print
    pass


def evaluate(data_source, data_order):
    random.shuffle(data_order)
    for batch_num in data_order:
        sent, ppos, pneg, field, value = get_data(data_source, batch_num)
        #TODO: model forward, loss calculation and debug print
    pass


best_val_loss = None
val_losses = []
train_losses = []
try:
    for epoch in range(1, total_epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(data_padded[2], val_batches)
        val_losses.append(val_loss)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.6f}s | valid loss {:5.6f} | '
              'valid ppl {:8.6f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(model_save_path+"best_model.pth", 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 2
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

plot(train_losses, val_losses, plot_save_path)

# Load the best saved model.
with open(model_save_path+"best_model.pth", 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(data_padded[1], test_batches)
print('=' * 89)
print('| End of training | test loss {:5.6f} | test ppl {:8.6f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)