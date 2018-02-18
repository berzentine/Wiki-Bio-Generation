import torchwordemb
import argparse
import torch
import data as data
###############################################################################
# Parse command line arguments
###############################################################################

parser = argparse.ArgumentParser(description='PyTorch Text Generation Model')
parser.add_argument('--cuda', action='store_true', default=True, help='use CUDA')
parser.add_argument('--seed', type=int, default=1,help='random seed')
parser.add_argument('--batchsize', type=int, default=32,help='batchsize')
parser.add_argument('--data', type=str, default='./data/Wiki-Data/wikipedia-biography-dataset/',help='location of the data corpus')
args = parser.parse_args()
cuda = args.cuda
seed = args.seed
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
# Batchify
# TODO: Padding
# TODO: Stacking
###############################################################################
def batchify(corpus):
    generate = 1 # generates only the top k sentence of the biography

    train_box = [] # List of tokensized sequences
    train_sent = []
    val_box = []
    val_sent = []
    test_box = []
    test_sent = []
    for example in corpus.train:
        temp = []
        for index in example.box.split('\t'):
            for i in index.split(':'):
                temp.append(i)
        train_box.append(temp)

        temp = []
        for index in range(0,len(example.sent.split('\n'))):
            if index<=generate-1:
                for i in example.sent.split('\n')[index].split():
                    temp.append(i)
        train_sent.append(temp)

    for example in corpus.val:
        temp = []
        for index in example.box.split('\t'):
            for i in index.split(':'):
                temp.append(i)
        val_box.append(temp)

        temp = []
        for index in range(0,len(example.sent.split('\n'))):
            if index<=generate-1:
                for i in example.sent.split('\n')[index].split():
                    temp.append(i)
        val_sent.append(temp)

    for example in corpus.test:
        temp = []
        for index in example.box.split('\t'):
            for i in index.split(':'):
                temp.append(i)
        test_box.append(temp)

        temp = []
        for index in range(0,len(example.sent.split('\n'))):
            if index<=generate-1:
                for i in example.sent.split('\n')[index].split():
                    temp.append(i)
        test_sent.append(temp)

    return train_box, train_sent, val_box, val_sent, test_box, test_sent
#def pad_collate(train_box, train_sent, val_box, val_sent, test_box, test_sent):


train_box, train_sent, val_box, val_sent, test_box, test_sent = Batchify(corpus)
#train_box, train_sent, val_box, val_sent, test_box, test_sent = pad_collate(train_box, train_sent, val_box, val_sent, test_box, test_sent)

###############################################################################
# TODO: Build Model
# Train Model
# Test Model
###############################################################################
