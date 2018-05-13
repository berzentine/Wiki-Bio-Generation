#with open('Wiki-Data/wikipedia-biography-dataset/train/train.sent')
file = open("Wiki-Data/wikipedia-biography-dataset/train/train.sent", "r")
sentences = [line.split('\n')[0] for line in file]
file = open("Wiki-Data/wikipedia-biography-dataset/train/train.nb", "r")
no_sentences = [int(line.split('\n')[0]) for line in file]
z = 0
words = dict()
for s in range(len(no_sentences)):
    current = sentences[z:z + no_sentences[s]]
    z = z + no_sentences[s]
    current = current[0:1]
    temp_sent = []
    temp_sent_ununk = []
    for c in current:
        c = c.split(' ')
        for word in c:
            if word not in words:
                words[word]=0
            words[word]+=1
words_sorted = sorted(words.items(), key=lambda x: x[1], reverse=True)
with open('full_train_vocab_non5.txt', 'w') as wp:
    for w in words_sorted:
        if w[1]>5:
            wp.write(w[0]+"\t"+str(w[1])+'\n')
