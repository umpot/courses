import pickle
import numpy as np
from keras import Input
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

glove_fp = '../../../data/glove/glove.6B/glove.6B.50d.txt'
def load_glove(fp):
    with open(fp) as f:
        ll = f.readlines()
        ll = [x.split() for x in ll]
        words = [x[0] for x in ll]
        vecs = [x[1:] for x in ll]
        vecs = [[float(y) for y in x] for x in vecs]
        return {words[i]: vecs[i] for i in range(len(words))}

def get_sum_emb_input(words_arr, emb_dict):
    res = []
    for x in words_arr:
        res_inner = []
        for y in x:
            try:
                res_inner.append(emb_dict[y])
            except:
                print y

        res_inner = np.mean(res_inner, axis=0)
        res.append(res_inner)

    return res

def get_merge_emb_input(words_arr, emb_dict, max_num):
    res = []
    for x in words_arr:
        res_inner = []
        for y in x:
            try:
                res_inner+=emb_dict[y]
            except:
                print y
        if len(res_inner)>=max_num:
            res.append(res_inner[:max_num])
        else:
            delta = max_num-len(res_inner)
            res_inner=res_inner+[0]*delta
            res.append(res_inner)

    return res

def shuffle_input(x,y):
    z = zip(x,y)
    z = np.random.permutation(z)

    return np.array([m[0] for m in z]), np.array([m[1] for m in z])


word_to_index = imdb.get_word_index()

index_to_word = {v: k for k, v in word_to_index.iteritems()}

path = get_file('imdb_full.pkl',
                origin='https://s3.amazonaws.com/text-datasets/imdb_full.pkl',
                md5_hash='d091312047c43cf9e4e38fef92437263')
f = open(path, 'rb')
(x_train, labels_train), (x_test, labels_test) = pickle.load(f)
x_train_words = [[index_to_word[y] for y in x] for x in x_train]
x_test_words = [[index_to_word[y] for y in x] for x in x_test]

x_train_words, labels_train = shuffle_input(x_train_words, labels_train)
x_test_words, labels_test = shuffle_input(x_test_words, labels_test)

d_glove = load_glove(glove_fp)
embeding_dim=50

train_inp = get_merge_emb_input(x_train_words, d_glove, embeding_dim*200)
test_inp = get_merge_emb_input(x_test_words, d_glove, embeding_dim*200)

# train_inp = get_sum_emb_input(x_train_words, d_glove)#, 200*embeding_dim
# test_inp = get_sum_emb_input(x_test_words, d_glove)

train_inp = np.array(train_inp)
test_inp = np.array(test_inp)




model = Sequential([
    Dense(1, activation='sigmoid', input_shape=(embeding_dim,)),
])

# model = Sequential([
#     Dense(1, activation='sigmoid', input_shape=(embeding_dim*50,)),
#     Dropout(0.2)
# ])
#
# model = Sequential([
#     Dense(10, activation='relu', input_shape=(embeding_dim,)),
#     Dropout(0.2),
#     Dense(1, activation='sigmoid')
# ])
#
#
model = Sequential([
    Dense(200, activation='relu', input_shape=(embeding_dim*200,)),
    Dropout(0.3),
    Dense(50, activation='relu', input_shape=(embeding_dim*200,)),
    Dropout(0.4),
    Dense(10, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_inp, labels_train, validation_data=(test_inp, labels_test), epochs=10, batch_size=64)

# train = np.random.uniform(0,1, 1000)
# labels = [int(x>0.5) for x in train]
#
# model = Sequential([
#     Dense(1, activation='sigmoid', input_dim=1)
# ])
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(train, labels, nb_epoch=1, batch_size=64)
