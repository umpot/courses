import pickle
import numpy as np
from keras import Input
from keras.callbacks import Callback

from keras.datasets import imdb
from keras.engine import Model
from keras.layers import Embedding, Dense, Flatten, Dropout, SpatialDropout1D, Convolution1D, MaxPooling1D, merge, Merge
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.utils import get_file
from sklearn.metrics import roc_auc_score
import os
import bcolz
import re

glove_fp = '../../../data/glove/glove.6B/glove.6B.50d.txt'
embeding_dim=50



def load_train_test_raw():
    path = get_file('imdb_full.pkl',
                    origin='https://s3.amazonaws.com/text-datasets/imdb_full.pkl',
                    md5_hash='d091312047c43cf9e4e38fef92437263')
    f = open(path, 'rb')
    return pickle.load(f)


def preprocess_for_nn(x, vocab_size, seq_len):
    x= [np.array([i if i < vocab_size - 1 else vocab_size - 1 for i in s]) for s in x]
    return sequence.pad_sequences(x, maxlen=seq_len, value=0)

def load_glove(fp):
    with open(fp) as f:
        ll = f.readlines()
        ll = [x.split() for x in ll]
        words = [x[0] for x in ll]
        vecs = [x[1:] for x in ll]
        vecs = [[float(y) for y in x] for x in vecs]
        return {words[i]: vecs[i] for i in range(len(words))}

def shuffle_input(x,y):
    z = zip(x,y)
    z = np.random.permutation(z)

    return np.array([m[0] for m in z]), np.array([m[1] for m in z])

def create_emb_matrix(d_glove, index_to_word, vocab_size):
    res = np.zeros((vocab_size, embeding_dim))
    zero = lambda : np.random.normal(scale=0.6, size=(embeding_dim,))
    for i in range(1, vocab_size):
        word = index_to_word[i]
        vec = d_glove.get(word, None)
        if vec is None:
            try:
                print 'Cant find {} in glove'.format(word)
            except:
                print 'Blja', word
            vec = zero()

        res[i] = vec

    res[-1] = zero()
    return res/3

vocab_size = 5000
seq_len = 500

word_to_index = imdb.get_word_index()
index_to_word = {v: k for k, v in word_to_index.iteritems()}

d_glove = load_glove(glove_fp)
emb_matrix = create_emb_matrix(d_glove,index_to_word, vocab_size)

(x_train, labels_train), (x_test, labels_test) = load_train_test_raw()
trn = preprocess_for_nn(x_train, vocab_size, seq_len)
test = preprocess_for_nn(x_test, vocab_size, seq_len)

inp = Input(shape=(vocab_size, embeding_dim))

embeding_layer = Embedding(
    input_dim=vocab_size,
    output_dim=embeding_dim,
    input_length=seq_len,
    weights=[emb_matrix],
    trainable=False,
    dropout=0.2
)

conv_layers=[]
for sz in [3,4,5]:
    conv = Convolution1D(
        filters=64,
        kernel_size=sz,
        border_mode='same',
        activation='relu')(inp)

    conv_layers.append(conv)

conv_layer = [MaxPooling1D()(x) for x in conv_layers]

flattened = [Flatten()(x) for x in conv_layers]
merged = Merge(mode='concat')(flattened)

graph = Model(inp, merged)

emb_model = Sequential([
    embeding_layer,
    Dropout (0.2),
    graph,
    Dropout(0.5),
    Dense(100, activation='relu'),
    Dropout(0.7),
    Dense(1, activation='sigmoid')])

emb_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
emb_model.summary()

emb_model.fit(trn,
              labels_train,
              validation_data=(test, labels_test),
              nb_epoch=1,
              batch_size=64)


