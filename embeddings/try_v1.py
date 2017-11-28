import pickle
import numpy as np
from keras.callbacks import Callback

from keras.datasets import imdb
from keras.layers import Embedding, Dense, Flatten, Dropout, SpatialDropout1D, Convolution1D, MaxPooling1D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.utils import get_file
from sklearn.metrics import roc_auc_score
import os
import bcolz
import re


class roc_callback(Callback):
    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc, 4)), str(round(roc_val, 4))))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


idx = imdb.get_word_index()

idx_arr = sorted(idx, key=idx.get)
idx2word = {v: k for k, v in idx.iteritems()}

path = get_file('imdb_full.pkl',
                origin='https://s3.amazonaws.com/text-datasets/imdb_full.pkl',
                md5_hash='d091312047c43cf9e4e38fef92437263')
f = open(path, 'rb')
(x_train, labels_train), (x_test, labels_test) = pickle.load(f)

vocab_size = 5000

trn = [np.array([i if i < vocab_size - 1 else vocab_size - 1 for i in s]) for s in x_train]
test = [np.array([i if i < vocab_size - 1 else vocab_size - 1 for i in s]) for s in x_test]

lens = np.array(map(len, trn))
(lens.max(), lens.min(), lens.mean())

seq_len = 500
trn = sequence.pad_sequences(trn, maxlen=seq_len, value=0)
test = sequence.pad_sequences(test, maxlen=seq_len, value=0)


def sentence_to_input(s):
    seq = [idx.get(x, vocab_size - 1) for x in s.split()]
    seq = [x if x < vocab_size else vocab_size - 1 for x in seq]
    return sequence.pad_sequences([seq], maxlen=seq_len, value=0)


def predict_model(ss, model):
    return model.predict(sentence_to_input(ss))


model = Sequential([
    Embedding(vocab_size, 10, input_length=seq_len),
    Flatten(),
    Dense(100, activation='relu'),
    Dropout(0.7),
    Dense(1, activation='sigmoid')])

model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.summary()

model.fit(trn,
          labels_train,
          validation_data=(test, labels_test),
          nb_epoch=1,
          batch_size=64,
          callbacks=[roc_callback(training_data=(trn, labels_train),
                                  validation_data=(test, labels_test))])

conv1 = Sequential([
    Embedding(vocab_size, 32, input_length=seq_len),
    SpatialDropout1D(rate=0.2),
    Dropout(0.2),
    Convolution1D(64, 5, border_mode='same', activation='relu'),
    Dropout(0.2),
    MaxPooling1D(),
    Flatten(),
    Dense(100, activation='relu'),
    Dropout(0.7),
    Dense(1, activation='sigmoid')
])

conv1.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
conv1.summary()

conv1.fit(trn,
          labels_train,
          validation_data=(test, labels_test),
          nb_epoch=2,
          batch_size=64,
          callbacks=[roc_callback(training_data=(trn, labels_train),
                                  validation_data=(test, labels_test))])

glove_fp = '../../../data/glove/glove.6B/glove.6B.50d.txt'
def load_glove(fp):
    with open(fp) as f:
        ll = f.readlines()
        ll = [x.split() for x in ll]
        words = [x[0] for x in ll]
        vecs = [x[1:] for x in ll]
        vecs = [[float(y) for y in x] for x in vecs]
        return {words[i]: vecs[i] for i in range(len(words))}


d_glove = load_glove(glove_fp)
embeding_dim=50

def create_emb_matrix():
    res = np.zeros((vocab_size, embeding_dim))
    zero = lambda : np.random.normal(scale=0.6, size=(embeding_dim,))
    for i in range(1, vocab_size):
        word = idx2word[i]
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

emb_matrix = create_emb_matrix()


blja = Sequential([
    Embedding(vocab_size, 50, input_length=seq_len, weights=[emb_matrix], trainable=False)
])

blja.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

def get_glove_dataset(dataset):
    """Download the requested glove dataset from files.fast.ai
    and return a location that can be passed to load_vectors.
    """
    # see wordvectors.ipynb for info on how these files were
    # generated from the original glove data.
    md5sums = {'6B.50d': '8e1557d1228decbda7db6dfd81cd9909',
               '6B.100d': 'c92dbbeacde2b0384a43014885a60b2c',
               '6B.200d': 'af271b46c04b0b2e41a84d8cd806178d',
               '6B.300d': '30290210376887dcc6d0a5a6374d8255'}
    glove_path = os.path.abspath('data/glove/results')
    return get_file(dataset,
                    'http://files.fast.ai/models/glove/' + dataset + '.tgz',
                    cache_subdir=glove_path,
                    md5_hash=md5sums.get(dataset, None),
                    untar=True)

def load_array(fname):
    return bcolz.open(fname)[:]

def load_vectors(loc):
    return (load_array(loc+'.dat'),
            pickle.load(open(loc+'_words.pkl','rb')),
            pickle.load(open(loc+'_idx.pkl','rb')))

vecs, words, wordidx = load_vectors(get_glove_dataset('6B.50d'))

def create_emb():
    n_fact = vecs.shape[1]
    emb = np.zeros((vocab_size, n_fact))

    for i in range(1,len(emb)):
        word = idx2word[i]
        if word and re.match(r"^[a-zA-Z0-9\-]*$", word):
            src_idx = wordidx[word]
            emb[i] = vecs[src_idx]
        else:
            # If we can't find the word in glove, randomly initialize
            emb[i] = np.random.normal(scale=0.6, size=(n_fact,))

    # This is our "rare word" id - we want to randomly initialize
    emb[-1] = np.random.normal(scale=0.6, size=(n_fact,))
    emb/=3
    return emb

emb = create_emb()

emb_model = Sequential([
    Embedding(vocab_size, 50, input_length=seq_len, dropout=0.2,
              weights=[emb], trainable=False),
    Dropout(0.25),
    Convolution1D(64, 3, border_mode='same', activation='relu'),
    Dropout(0.25),
    MaxPooling1D(),
    Flatten(),
    Dense(100, activation='relu'),
    Dropout(0.7),
    Dense(1, activation='sigmoid')])



emb_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
emb_model.summary()

emb_model.fit(trn,
              labels_train,
              validation_data=(test, labels_test),
              nb_epoch=2,
              batch_size=64)