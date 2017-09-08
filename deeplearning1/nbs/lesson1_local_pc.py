from __future__ import division,print_function

import os, json
from glob import glob
from time import strftime, gmtime

import numpy as np
import pandas as pd
import sys

import re
from keras import backend
backend.set_image_dim_ordering('tf')

current_dir = '/home/dpetrovskyi/PycharmProjects/courses/deeplearning1/nbs'
sys.path.append(current_dir)
import utils; reload(utils)
from utils import plots
import vgg16; reload(vgg16)
from vgg16 import Vgg16



base_path = "/home/dpetrovskyi/fai"
train_path = base_path + '/sample'
valid_path = base_path + '/valid'
test_path = base_path + '/to_predict'

batch_size = 4
vgg = Vgg16()

def get_time_str():
    return strftime("%Y_%m_%d__%H_%M_%S", gmtime())

p=re.compile('\d+')
def extract_id(f_name):
    return p.search(f_name).group()


def get_list_of_images(d):
    b_sz = 4
    bb = vgg.get_batches(d, shuffle=False, batch_size=b_sz)

    res = []
    started = False

    counter = 0
    while True:
        print(counter)
        counter+=1

        y = bb.next()
        res+=list(y[0])
        if started and bb.batch_index==0:
            break
        started = True

    return np.array(res), bb.filenames

def create_submission(res, filenames):
    ids = [extract_id(s) for s in filenames]
    probs = res[:,1]
    z = zip(ids, probs)
    z.sort(key=lambda s: int(s[0]))
    df = pd.DataFrame({'id':[x[0] for x in z], 'label':[x[1] for x in z]})
    f_name = 'sub_{}.csv'.format(get_time_str())
    df.to_csv(f_name, index=False)



train_batches = vgg.get_batches(train_path, batch_size=batch_size)
val_batches = vgg.get_batches(valid_path, batch_size=batch_size * 2)
to_predict_arr, filenames = get_list_of_images(test_path)


vgg.finetune(train_batches)
# vgg.fit(train_batches, val_batches, nb_epoch=10)
vgg.fit_without_val_batches(train_batches, nb_epoch=5)
print('Fitting is done')
res = vgg.predict_straight(to_predict_arr)
create_submission(res, filenames)
print("================Done!====================")