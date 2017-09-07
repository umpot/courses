from __future__ import division,print_function

import os, json
from glob import glob
import numpy as np
import sys

from keras import backend
backend.set_image_dim_ordering('th')

current_dir = '/home/dpetrovskyi/PycharmProjects/courses/deeplearning1/nbs'
sys.path.append(current_dir)
import utils; reload(utils)
from utils import plots
import vgg16; reload(vgg16)
from vgg16 import Vgg16



data_path = "/home/dpetrovskyi/fai"

batch_size = 4
vgg = Vgg16()

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

    return res

batches = vgg.get_batches(data_path + '/sample', batch_size=batch_size)
# imgs,labels = next(batches)
vgg.finetune(batches)
val_batches = vgg.get_batches(data_path+'/valid', batch_size=batch_size*2)

bl = vgg.fit(batches, val_batches, nb_epoch=1)
print('Fitting is done')


to_predict_batches = vgg.get_batches(data_path + '/to_predict', batch_size=batch_size)
res = vgg.predict(to_predict_batches)