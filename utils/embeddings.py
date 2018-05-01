import numpy as np
import gensim
from gensim.models import KeyedVectors


def filter_word_embeddings(vocab_size, i2w, eng_emb_path):
    word_vectors_eng = KeyedVectors.load_word2vec_format(eng_emb_path, binary=True)
    vec = np.zeros((vocab_size, word_vectors_eng.vector_size))
    count_eng = 0
    for i in range(0, vocab_size):
        word = i2w[i]
        if word in word_vectors_eng:
            vec[i] = word_vectors_eng[word]
        else:
            count_eng += 1
            vec[i] = word_vectors_eng['UNK']
    print("Count of UNK in Eng: "+str(count_eng))
    return vec
