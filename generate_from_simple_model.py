import argparse

import torch
from torch.autograd import Variable

import data_handler as data_handler

parser = argparse.ArgumentParser(description='PyTorch Text Generation Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/Wiki-Data/wikipedia-biography-dataset/',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./saved_models/',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--max_length', type=int, default='30',
                    help='max seq to generate')
parser.add_argument('--samples_to_generate', type=int, default='10',
                    help='no: of biographies to generate to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

SOS_token = "<sos>"
EOS_token = "<eos>"
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

# if args.temperature < 1e-3:
#     parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint+"best_encoder.pth", 'rb') as f:
    encoder = torch.load(f)
encoder.eval()
with open(args.checkpoint+"best_decoder.pth", 'rb') as f:
    decoder = torch.load(f)
decoder.eval()

if args.cuda:
    encoder.cuda()
    decoder.cuda()

print("Load data")
corpus = data_handler.Corpus(args.data)
#Use: table: corpus.train[i].box, bio: corpus.train[i].sent
#Sentence are \n seprated and box is \t seprated
print('Train:', len(corpus.train), 'Validation:', len(corpus.val), 'Test:', len(corpus.test))



vocab_size = len(corpus.dictionary)


#if args.cuda:
#    input.data = input.cuda()

def get_table(i):
    temp = []
    for index in corpus.train[i].box.split('\t'):
        for i in index.split(':'):
            temp.append(int(i))
    return temp


with open(args.outf, 'w') as outf:
    for n in range(args.samples_to_generate):
        encoder_hidden = encoder.init_hidden(1)
        table = Variable(torch.LongTensor([get_table(n)]), volatile=True)
        if args.cuda:
            table = table.cuda()
        table_encoded, table_hidden = encoder.forward(table, encoder_hidden)
        hidden = (table_hidden[0].view(-1, table_hidden[0].size(0)*table_hidden[0].size(2)).unsqueeze(0), table_hidden[1].view(-1, table_hidden[1].size(0)*table_hidden[1].size(2)).unsqueeze(0) )

        decoder_input = Variable(torch.LongTensor([corpus.dictionary.word2idx[SOS_token]]), volatile = True)
        if args.cuda:
            decoder_input = decoder_input.cuda()
        # print decoder_input.unsqueeze(0).unsqueeze(0)
        outf.write("Table: \n")
        for i in get_table(n):
            outf.write(corpus.dictionary.idx2word[i] + ('\n' if i % 20 == 19 else ' '))
        outf.write('\nBiography: \n')
        for i in range(args.max_length):
            output, hidden = decoder(decoder_input, hidden, True)
            word_weights = output.squeeze().data.exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            decoder_input.data.fill_(word_idx)
            word = corpus.dictionary.idx2word[word_idx]
            outf.write(word + ('\n' if i % 20 == 19 else ' '))
            if word_idx == corpus.dictionary.word2idx[EOS_token]:
                break
        outf.write("\n\n")
        outf.write('=' * 89)
