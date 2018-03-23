# should take test data as input
# spit out the generated text and the original text
from batchify_pad import batchify

corpus = data_reader.Corpus(data_path, vocab_path, 1, limit, verbose)
WORD_VOCAB_SIZE = len(corpus.word_vocab)
FIELD_VOCAB_SIZE = len(corpus.field_vocab)
POS_VOCAB_SIZE = len(corpus.pos_vocab)

def generate(reference_sent, data_box):
    corpus.test_value, corpus.test_value_len, corpus.test_field, corpus.test_field_len, corpus.test_ppos, corpus.test_ppos_len, \
    corpus.test_pneg, corpus.test_pneg_len, corpus.test_sent, corpus.test_sent_len = batchify([corpus.test_value, corpus.test_field , corpus.test_ppos, corpus.test_pneg, corpus.test_sent], 1, verbose)
    
    total_loss = total_words = 0
    model.eval()
    start_time = time.time()
    random.shuffle(data_order)
    for batch_num in data_order:
        sent, sent_len, ppos, pneg, field, value, target = get_data(data_source, batch_num, True)
        gen_seq = model.generate(value, field, ppos, pneg, 1, False, max_length, \
                                                corpus.word_vocab.word2idx["<sos>"],  corpus.word_vocab.word2idx["<eos>"], corpus.word_vocab)
    return data_sent # TODO
