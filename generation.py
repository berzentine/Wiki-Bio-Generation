# should take test data as input
# spit out the generated text and the original text
def (data_source, data_order, test):
    total_loss = total_words = 0
    model.eval()
    start_time = time.time()
    random.shuffle(data_order)
    for batch_num in data_order:
        sent, sent_len, ppos, pneg, field, value, target = get_data(data_source, batch_num, True)
        gen_seq = model.generate(value, field, ppos, pneg, 1, False, max_length, \
                                                corpus.word_vocab.word2idx["<sos>"],  corpus.word_vocab.word2idx["<eos>"], corpus.word_vocab)


    return 0 # TODO
