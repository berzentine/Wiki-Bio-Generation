
def EditDistance(box_word, sent_word):
    import Levenshtein
    return float(Levenshtein.ratio('hello world', 'hello'))

def levenshtein(source, target):
    import numpy as np
    if len(source) < len(target):
        return levenshtein(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]

# Usage: print levenshtein('aa','bb')
boxes = []
# cut off at 100
# remove None's <none>

#Read data
path = ''
with open(path+'train/train.box','r') as boxfile:
    for lines in boxfile:
        each = lines.split('\n')[0].replace('\t',' ')
        keys = each.split(' ')
        box = []
        for k in keys:
            if k.split(':')[1]!='<none>':
                box.append(k.split(':')[1])
            if len(box)>=100:
                break
        boxes.append(' '.join(box))

#Read data
with open(path+'train/train.nb','r') as nbfile:
    nblines = nbfile.readlines()

with open(path+'train/train.sent','r') as sentfile:
    sentlines = sentfile.readlines()

start = 0
index = 0
sentences = []
for nb in nblines:
    sentences.append(sentlines[start].split('\n')[0])
    skip = int(nblines[index].split('\n')[0])
    index+=1
    start=start+skip


# generating using nltk model
from nltk.translate import AlignedSent
from nltk.translate import IBMModel1
from nltk.translate import Alignment
bitext = []
for i in range(len(boxes)):
    bitext.append(AlignedSent(boxes[i].split(), sentences[i].split()))

print('Doing alignments generation')
ibm1 = IBMModel1(bitext, 5)

print('Doing translation files')
diction = ibm1.translation_table
for d in diction:
    # d is table word, diction[d] has sent word and the P(sent word| table word)
    for t in diction[d]:
        print d, t, diction[d][t]

# generate using fastalign
"""
distance_dict = {}
for i in range(len(boxes)):
    #print boxes[i], '|||',sentences[i]
    box_words = boxes[i].split()
    sent_words = sentences[i].split()
    for b in range(len(box_words)):
        for s in range(len(sent_words)):
            if box_words[b] not in distance_dict:
                distance_dict[box_words[b]] = {}
            if sent_words[s] not in distance_dict[box_words[b]]:
                distance_dict[box_words[b]][sent_words[s]] = levenshtein(box_words[b], sent_words[s])

            #print , box_words[b], sent_words[s]
for k in distance_dict.keys():
    for s in distance_dict[k]:
        print k, s, distance_dict[k][s]"""
