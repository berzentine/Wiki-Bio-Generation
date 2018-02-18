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
def pad_collate(train_box, train_sent, val_box, val_sent, test_box, test_sent):
    
    return train_box, train_sent, val_box, val_sent, test_box, test_sent
