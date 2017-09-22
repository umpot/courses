from keras import backend as K
from keras.utils.conv_utils import convert_kernel
import tensorflow as tf
from time import strftime, gmtime

import numpy as np
import pandas as pd
import re

from keras.layers.core import Lambda, Dense, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from keras.optimizers import Adam, RMSprop

vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1))

def vgg_preprocess(x):
    x = x-vgg_mean

    return x[:, ::-1] # reverse axis rgb->bgr


ops = []
weights_fp = "/home/dpetrovskyi/.keras/models/vgg16.h5"
tf_weigths_fp = "/home/dpetrovskyi/.keras/models/vgg16_tf.h5"

def build_vgg_model(weights_fp):
    model = Sequential()
    model.add(Lambda(vgg_preprocess, input_shape=(3, 224, 224), output_shape=(3, 224, 224)))

    number_of_filters = 64

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(number_of_filters, (3, 3), activation="relu"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(number_of_filters, (3, 3), activation="relu"))

    model.add(MaxPooling2D((2,2), strides=(2,2)))

    number_of_filters = 128

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(number_of_filters, (3, 3), activation="relu"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(number_of_filters, (3, 3), activation="relu"))

    model.add(MaxPooling2D((2,2), strides=(2,2)))

    number_of_filters = 256

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(number_of_filters, (3, 3), activation="relu"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(number_of_filters, (3, 3), activation="relu"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(number_of_filters, (3, 3), activation="relu"))

    model.add(MaxPooling2D((2,2), strides=(2,2)))

    number_of_filters = 512

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(number_of_filters, (3, 3), activation="relu"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(number_of_filters, (3, 3), activation="relu"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(number_of_filters, (3, 3), activation="relu"))

    model.add(MaxPooling2D((2,2), strides=(2,2)))

    number_of_filters = 512

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(number_of_filters, (3, 3), activation="relu"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(number_of_filters, (3, 3), activation="relu"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(number_of_filters, (3, 3), activation="relu"))

    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())

    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(1000, activation="softmax"))

    model.load_weights(weights_fp)

    return model

model = build_vgg_model(weights_fp)
for layer in model.layers:
    if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D', 'Convolution3D', 'AtrousConvolution2D']:
        original_w = K.get_value(layer.W)
        converted_w = convert_kernel(original_w)
        ops.append(tf.assign(layer.W, converted_w).op)


K.get_session().run(ops)

model.save_weights(tf_weigths_fp)