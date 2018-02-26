import torchwordemb
import math
import time
import argparse
import torch
import random
import torch.nn as nn
import torch.optim as optim
import new_data_handler as data_handler
from batchify_pad import batchify, pad_collate
from models.LSTMDecoder import LSTMDecoder
from models.BiLSTMEncoder import BiLSTMEncoder

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
parser.add_argument('--lr', type=float, default=0.1,help='learning rate')
parser.add_argument('--data', type=str, default='./data/Wiki-Data/wikipedia-biography-dataset/',help='location of the data corpus')
parser.add_argument('--model_save_path', type=str, default='./saved_models/',help='location of the best model to save')
parser.add_argument('--plot_save_path', type=str, default='./saved_models/loss_plot.png',help='location of the loss plot to save')
parser.add_argument('--emsize', type=int, default=200,help='size of word embeddings')
parser.add_argument('--nlayers', type=int, default=1,help='number of layers')
parser.add_argument('--nhid', type=int, default=100,help='number of hidden units per layer')
parser.add_argument('--dropout', type=float, default=0.2,help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--clip', type=float, default=0.2,help='gradient clip')
parser.add_argument('--teacher_forcing_ratio', type=float, default=0.4,help='gradient clip')
parser.add_argument('--log_interval', type=int, default=500,help='log interval')
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
teacher_forcing_ratio = args.teacher_forcing_ratio
data_path = args.data
plot_save_path = args.plot_save_path
model_save_path = args.model_save_path
lr = args.lr
clip = args.clip
log_interval = args.log_interval
SOS_token = "<sos>"
EOS_token = "<eos>"
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


def get_targets(source, box_batch, evaluation=False):
    batch = source

    data = Variable(torch.LongTensor([corpus.dictionary.word2idx[SOS_token]]*box_batch.size(0)), volatile = evaluation)
    target = batch[:, 1:batch.size(1)]
    data = torch.stack(data)
    target = torch.stack(target)
    if cuda:
        data = data.cuda()
    if cuda:
        target = target.cuda()
    if cuda:
        box_batch = box_batch.cuda()
    # data = Variable(data, volatile=evaluation)
    target = Variable(target)
    box_batch = Variable(box_batch)
    return data, target, box_batch


vocab_size = len(corpus.dictionary)
decoder = LSTMDecoder(dropout=dropout, vocab_size=vocab_size, embed_size=emb_size, num_layers=num_layers, decoder_hidden_size=hidden_size)
encoder = BiLSTMEncoder(dropout=dropout, vocab_size=vocab_size, embed_size=emb_size, num_layers=num_layers, encoder_hidden_size=hidden_size)
if args.cuda:
    decoder.cuda()
    encoder.cuda()


criterion = nn.CrossEntropyLoss()
encoder_optimizer = optim.SGD(filter(lambda p: p.requires_grad, encoder.parameters()), lr=lr)
decoder_optimizer = optim.SGD(filter(lambda p: p.requires_grad, decoder.parameters()), lr=lr)


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def train():
    #TODO random shuffle
    encoder.train()
    decoder.train()
    start_time = time.time()
    total_loss = 0
    total_words = 0
    batch_loss = batch_words = i = 0
    for box_batch, sent_batch, box_length_batch, sent_length_batch in zip(train_box, train_sent, train_box_length, train_sent_length):
        i+=1
        data, targets, box_batch = get_targets(sent_batch, box_batch, False)
        initial_encoder_hidden = encoder.init_hidden(list(box_batch.size())[0])
        table_encoded, table_hidden = encoder.forward(box_batch, initial_encoder_hidden)
        hidden = (table_hidden[0].view(-1, table_hidden[0].size(0)*table_hidden[0].size(2)).unsqueeze(0), table_hidden[1].view(-1, table_hidden[1].size(0)*table_hidden[1].size(2)).unsqueeze(0) )

        target_length = targets.size(1)
        loss = 0
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            for di in range(target_length):
                # print hidden
                output, hidden = decoder.forward(data, hidden, False)
                best_vocab, best_index = output.data.topk(1)
                next_index = best_index[:,0]
                # print next_index
                # exit(0)
                decoder_input = Variable(next_index)
                data = targets[:, di].unsqueeze(1)
                output = output.squeeze()
                loss += criterion(output, targets[:, di])
        else:
            for di in range(target_length):
                # print hidden
                output, hidden = decoder.forward(data, hidden, False)
                best_vocab, best_index = output.data.topk(1)
                next_index = best_index[:,0]
                # print next_index
                # exit(0)
                decoder_input = Variable(next_index)
                data = decoder_input.cuda() if cuda else decoder_input
                output = output.squeeze()
                loss += criterion(output, targets[:,di])
                # if next_index == corpus.dictionary.word2idx[EOS_token]:
                #     break
        # output = model.forward(data, box_batch, initial_lm_hidden, initial_encoder_hidden, False)
        # loss = criterion(output.view(-1, vocab_size), targets.view(-1))
        total_loss += loss.data
        total_words += sum(sent_length_batch)
        batch_loss += loss.data
        batch_words += sum(sent_length_batch)
        #hidden = repackage_hidden(hidden)
        decoder_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
        decoder_optimizer.lr = lr
        decoder_optimizer.step()
        encoder_optimizer.lr = lr
        encoder_optimizer.step()
        if i % log_interval == 0 and i > 0:
            cur_loss = batch_loss[0] / batch_words
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.6f} | ppl {:8.6f}'.format(
                epoch, i, len(train_sent), lr, elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            batch_loss = 0
            batch_words = 0
            start_time = time.time()
        del sent_batch, box_batch, data, targets, output
    train_losses.append(total_loss[0]/total_words)


def evaluate(data_source_box, data_source_sent, data_source_box_lengths, data_source_sent_lengths):
    #TODO random shuffle
    encoder.eval()
    decoder.eval()
    start_time = time.time()
    total_loss = 0
    total_words = 0
    for box_batch, sent_batch, box_length_batch, sent_length_batch in zip(data_source_box, data_source_sent, data_source_box_lengths, data_source_sent_lengths):
        data, targets, box_batch = get_targets(sent_batch, box_batch, True)
        initial_encoder_hidden = encoder.init_hidden(list(box_batch.size())[0])
        table_encoded, table_hidden = encoder.forward(box_batch, initial_encoder_hidden)
        hidden = (table_hidden[0].view(-1, table_hidden[0].size(0)*table_hidden[0].size(2)).unsqueeze(0), table_hidden[1].view(-1, table_hidden[1].size(0)*table_hidden[1].size(2)).unsqueeze(0) )

        target_length = targets.size(1)
        loss = 0

        for di in range(target_length):
            # print hidden
            output, hidden = decoder.forward(data, hidden, False)
            best_vocab, best_index = output.data.topk(1)
            next_index = best_index[:,0]
            # print next_index
            # exit(0)
            decoder_input = Variable(next_index)
            data = decoder_input.cuda() if cuda else decoder_input
            output = output.squeeze()
            loss += criterion(output, targets[:,di])
            # if next_index == corpus.dictionary.word2idx[EOS_token]:
            #     break
        # output = model.forward(data, box_batch, initial_lm_hidden, initial_encoder_hidden, False)
        # loss = criterion(output.view(-1, vocab_size), targets.view(-1))
        total_loss += loss.data
        total_words += sum(sent_length_batch)
    return total_loss[0]/total_words

best_val_loss = None
val_losses = []
train_losses = []
try:
    for epoch in range(1, total_epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_box, val_sent, val_box_length, val_sent_length)
        val_losses.append(val_loss)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.6f}s | valid loss {:5.6f} | '
              'valid ppl {:8.6f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(model_save_path+"best_encoder.pth", 'wb') as f:
                torch.save(encoder, f)
            with open(model_save_path+"best_decoder.pth", 'wb') as f:
                torch.save(decoder, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 2
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

plot(train_losses, val_losses, plot_save_path)

# Load the best saved model.
with open(model_save_path+"best_encoder.pth", 'rb') as f:
    encoder = torch.load(f)
with open(model_save_path+"best_decoder.pth", 'rb') as f:
    decoder = torch.load(f)

# Run on test data.
test_loss = evaluate(test_box, test_sent, test_box_length, test_sent_length)
print('=' * 89)
print('| End of training | test loss {:5.6f} | test ppl {:8.6f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)




