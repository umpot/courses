from __future__ import division,print_function

import os, json
from glob import glob
import numpy as np
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt
wd = '/home/dpetrovskyi/PycharmProjects/courses/deeplearning1/nbs'
import sys
sys.path.append(wd)

from keras import backend
backend.set_image_dim_ordering('th')


import utils; reload(utils)
from utils import plots

import vgg16; reload(vgg16)
from vgg16 import Vgg16


path = "/home/dpetrovskyi/fai"
vgg = Vgg16()
batches = vgg.get_batches(path+'/train', batch_size=4)
imgs,labels = next(batches)