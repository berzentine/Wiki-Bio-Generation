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
parser.add_argument('--data', type=str, default='./Wiki-Data/wikipedia-biography-dataset/',help='location of the data corpus')
args = parser.parse_args()
cuda = args.cuda
seed = args.seed
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
# TODO: Bachify
###############################################################################

###############################################################################
# TODO: Build Model
# Train Model
# Test Model
###############################################################################
