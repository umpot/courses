from __future__ import division,print_function
from keras import backend
backend.set_image_dim_ordering('th')
import os

path = "../../data/dogscats/"
print(os.getcwd())


import os, json
from glob import glob
import numpy as np
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt
print(os.getcwd())

import utils; reload(utils)
from utils import plots

batch_size=64

import vgg16; reload(vgg16)
from vgg16 import Vgg16

vgg = Vgg16()
# Grab a few images at a time for training and validation.
# NB: They must be in subdirectories named based on their category
batches = vgg.get_batches(path+'train', batch_size=batch_size)
val_batches = vgg.get_batches(path+'valid', batch_size=batch_size*2)
vgg.finetune(batches)
vgg.fit(batches, val_batches, nb_epoch=1)