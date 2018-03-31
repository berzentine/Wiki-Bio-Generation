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
from models.joint_model import Seq2SeqModel
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from models.ConditionedLM import ConditionedLM
from torch.autograd import Variable

###############################################################################
# Parse command line arguments
###############################################################################

parser = argparse.ArgumentParser(description='PyTorch Text Generation Model')
parser.add_argument('--cuda', action='store_true', default=False, help='use CUDA')
parser.add_argument('--verbose', action='store_true', default=False, help='use Verbose')
parser.add_argument('--limit', type=float, default=0.05,help='limit size of data')
parser.add_argument('--data', type=str, default='./data/Wiki-Data/wikipedia-biography-dataset/',help='location of the data corpus')
parser.add_argument('--vocab', type=str, default='./data/Wiki-Data/vocab/', help='location of the vocab files')
parser.add_argument('--model_save_path', type=str, default='./saved_models/best_model.pth',help='location of the best model to save')
parser.add_argument('--plot_save_path', type=str, default='./saved_models/loss_plot.png',help='location of the loss plot to save')
parser.add_argument('--max_sent_length', type=int, default=64,help='maximum sentence length for decoding')
parser.add_argument('--ref_path', type=str, required=True, help='Path for storing the reference file')
parser.add_argument('--gen_path', type=str, required=True, help='Path for storing the generated file')
parser.add_argument('--unk_gen_path', type=str, required=True, help='Path for storing the unk replaced file')
parser.add_argument('--beam_size', type=int, required=True, help='Beam size for performing beam search')
parser.add_argument('--seed', type=int, default=1,help='random seed')
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
data_path = args.data
seed = args.seed
vocab_path = args.vocab
plot_save_path = args.plot_save_path
model_save_path = args.model_save_path
max_length = args.max_sent_length
ref_path = args.ref_path
gen_path = args.gen_path
unk_gen_path = args.unk_gen_path
beam_size = args.beam_size
batch_size = 1
batchsize = 1
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
corpus.valid_pneg, corpus.valid_pneg_len, corpus.valid_sent, corpus.valid_sent_len, \
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

# Get the encoder state from the encoder
def getEncoder(value, field, ppos, pneg):
    input_d = model.sent_lookup(value)
    input_z = torch.cat((model.field_lookup(field), model.ppos_lookup(ppos), model.pneg_lookup(pneg)), 2)
    input = torch.cat((input_d,input_z), 2)
    encoder_output, encoder_hidden = model.encoder(input, None)
    prev_hidden =  (encoder_hidden[0].squeeze(0),encoder_hidden[1].squeeze(0))
    return encoder_output, encoder_hidden, prev_hidden

# get the decoder state from the decoder with a single call
def getDecoder(curr_input, prev_hidden, encoder_output):
    decoder_output, prev_hidden, attn_vector = model.decoder.forward(curr_input, prev_hidden, encoder_output)
    decoder_output = torch.nn.functional.softmax(decoder_output, dim = 2)
    decoder_output = model.linear_out(decoder_output)
    return decoder_output, attn_vector, prev_hidden

def getUNKrep(attn_vector, value_len, value_ununk, ununk_dictionary):
    unk_max_val, unk_max_idx = torch.max(attn_vector[0][0,:value_len[0]], 0)
    sub = value_ununk[0][unk_max_idx] # should be value_ununk
    word = ununk_dictionary.idx2word[int(sub)] # should be replaced from ununk dictionary word_ununk_vocab
    return word

def generate(value, value_len, field, ppos, pneg, batch_size, \
             train, max_length, start_symbol, end_symbol, dictionary, unk_symbol, \
             ununk_dictionary, value_ununk, value_mask, sent, beam):
    outputs, scores , hiddens, inputs, atts  = [], [], [], [], []
    candidates, candidate_unk_replaced = [[] for k in range(beam)], [[] for k in range(beam)]
    candidate_scores = [[] for k in range(beam)]

    encoder_output, encoder_hidden, prev_hidden = getEncoder(value, field, ppos, pneg)
    gen_seq, unk_rep_seq = [], []
    start_symbol =  Variable(torch.LongTensor(1,1).fill_(start_symbol))
    if cuda:
        start_symbol = start_symbol.cuda()
    curr_input = model.sent_lookup(start_symbol)
    # intialize the candidate scores and candidates to start with
    for j in range(beam):
        candidates[j].append(dictionary.idx2word[int(start_symbol)])
        candidate_scores[j].append(0)

    # Get the top K from timestep 0
    decoder_output, attn_vector, prev_hidden = getDecoder(curr_input, prev_hidden, encoder_output)
    values, indices = torch.topk(decoder_output, beam, 2)

    # for step time = 0 make beam_size updates
    for j in range(beam):
        outputs.append(indices[0,0,j].squeeze().data[0]) # what was the otput during this state
        scores.append(torch.log(values[0,0,j]).squeeze().data[0]) # what was the score of otput during this state
        hiddens.append(prev_hidden) # what was the produced hidden state for otput during this state
        inputs.append(curr_input) # what was the input during this state
        candidates[j].append(dictionary.idx2word[int(outputs[j][0])]) # update candidate vectors too with a + " "
        candidate_scores[j].append(scores[j][0])
        atts.append(attn_vector)
        if int(outputs[j][0]) == unk_symbol:
            if cuda: value_ununk = value_ununk.cuda()
            replacement = getUNKrep(attn_vector, value_len, value_ununk, ununk_dictionary)
            candidate_unk_replaced[j].append(replacement) # append a non UNK word here
        else:
            candidate_unk_replaced[j].append(dictionary.idx2word[int(outputs[j][0])])



    # for time step 1 onwards till max_length time step
    for t in range(1, max_length):
        temp_scores, temp_hiddens , temp_inputs, temp_outputs = [], [], [], [] # store K ones here for each jth exploration in outputs
        for j in range(beam): # explore outputs[j] which is
            curr_input = model.sent_lookup(Variable(torch.LongTensor(1,1).fill_(outputs[j])))
            decoder_output, attn_vector , prev_hidden = getDecoder(curr_input, hiddens[j], encoder_output)
            values, indices = torch.topk(torch.log(decoder_output)+scores[j], beam, 2)
            for p in range(beam): # append to temp_scores and all temp vectors the top k of outputs of [j]
                temp_outputs.append(indices[0,0,p].squeeze().data[0])
                temp_scores.append(values[0,0,j].squeeze().data[0])
                temp_hiddens.append(prev_hidden)
                temp_attention.append(attn_vector)
                temp_inputs.append(outputs[j])
        zipped = zip(temp_outputs, temp_scores, temp_hiddens, temp_inputs, temp_attention)
        zipped.sort(key = lambda t: t[1], reverse=True)
        outputs, scores , hiddens, inputs, attns = [], [], [], [], []
        for j in range(beam):
            outputs.append(zipped[j][0])
            scores.append(zipped[j][1])
            hiddens.append(zipped[j][2])
            inputs.append(zipped[j][3])
            atts.append(zipped[j][4])
            candidates[j].append(dictionary.idx2word[int(outputs[j])])
            candidate_scores[j].append(scores[j])
            if int(outputs[j][0]) == unk_symbol:
                if cuda: value_ununk = value_ununk.cuda()
                replacement = getUNKrep(atts[j], value_len, value_ununk, ununk_dictionary)
                candidate_unk_replaced[j].append(replacement) # append a non UNK word here
            else:
                candidate_unk_replaced[j].append(dictionary.idx2word[int(outputs[j][0])])

        #for j in range(beam): # update candidate vectors too with a + " "
    return candidates, candidate_unk_replaced, candidate_scores


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


val_batches = [x for x in range(0, len(corpus.valid["sent"]))]
test_batches = [x for x in range(0, len(corpus.test["sent"]))]
train_batches = [x for x in range(0, len(corpus.train["sent"]))]

def test_evaluate(data_source, data_order, test):
    gold_set = []
    pred_set = []
    pred_ununkset = []
    with open(ref_path, 'w') as rp:
        with open(gen_path, 'w') as gp:
            with open(unk_gen_path, 'w') as up:
                total_loss = total_words = 0
                model.eval()
                start_time = time.time()
                #random.shuffle(data_order)
                for batch_num in data_order:
                    sent, sent_len, ppos, pneg, field, value, value_len, target, actual_sent, sent_ununk, field_ununk , \
                    value_ununk, sent_mask, value_mask = get_data(data_source, batch_num, True)
                    ref_seq = []
                    for i in range(1, len(actual_sent[0])-1):
                        ref_seq.append(corpus.word_vocab.idx2word[int(actual_sent[0][i])])
                        #if WORD_VOCAB_SIZE>int(sent[0][i]):
                        #    ref_seq.append(corpus.word_vocab.idx2word[int(sent[0][i])])
                    candidates, candidate_unk_replaced, candidate_scores = generate(value, value_len, field, ppos, pneg, batchsize, False, max_length, \
                                                        corpus.word_vocab.word2idx["<sos>"],  corpus.word_vocab.word2idx["<eos>"], corpus.word_vocab, \
                                                        corpus.word_vocab.word2idx["UNK"], corpus.word_ununk_vocab, value_ununk, value_mask, actual_sent, beam_size)
                    gen_seq = []
                    unk_gen_seq = []

                    score  = [0 for k in range(beam_size)]
                    for k in range(beam_size):
                        for i in range(len(candidate_scores[k])):
                            score[k]+=candidate_scores[k][i]
                    zipped = zip(score, candidates, candidate_unk_replaced)
                    zipped.sort(key = lambda t: t[0], reverse=True)
                    gen_seq = zipped[0][1]
                    unk_gen_seq = zipped[0][2]

                    for r in ref_seq:
                        rp.write(r+" ")
                    for g in gen_seq:
                        gp.write(g+" ")
                    for u in unk_gen_seq:
                        up.write(u+" ")
                    rp.write("\n\n")
                    gp.write("\n\n")
                    up.write("\n\n")
                    gold_set.append(ref_seq)
                    pred_set.append(gen_seq)
                    pred_ununkset.append(unk_gen_seq)
    import os
    os.system("echo \"************ Python scores ************\"")
    bleu = corpus_bleu(gold_set, pred_set)
    print(bleu)
    os.system("echo \"************ Non-tokenized scores ************\"")
    os.system("./Scoring_scripts/multi-bleu.pl " +ref_path +" < " +gen_path +" | grep \"BLEU\"")
    os.system("echo \"======================================================\"")
    os.system("echo \"************ Tokenized scores ************\"")
    os.system("./Scoring_scripts/tokenizer.pl -l en < " +ref_path +" > "+ ref_path+".tokenized")
    os.system("./Scoring_scripts/tokenizer.pl -l en < " +gen_path +" > "+ gen_path+".tokenized")
    os.system("./Scoring_scripts/multi-bleu.pl "+ ref_path+".tokenized < "+gen_path+".tokenized | grep \"BLEU\"")
    #
    return

# Load the best saved model.
with open(model_save_path+"best_model.pth", 'rb') as f:
    model = torch.load(f)
# Run on test data.
test_evaluate(corpus.train, train_batches, test=True)
