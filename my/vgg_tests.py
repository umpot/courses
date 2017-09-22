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

import matplotlib.pyplot as plt

from keras import backend as K
K.set_image_dim_ordering("th")

vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1))

def vgg_preprocess(x):
    x = x-vgg_mean

    return x[:, ::-1] # reverse axis rgb->bgr

def plot_img(img):
    img = img.astype(np.uint8)
    img = np.transpose(img, (1,2,0))
    plt.imshow(img)

def get_time_str():
    return strftime("%Y_%m_%d__%H_%M_%S", gmtime())

p=re.compile('\d+')

def extract_id(f_name):
    return p.search(f_name).group()

def get_list_of_images(d):
    b_sz = 64
    bb = load_generator(d, b_sz)

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

def load_generator(fp, batch_size=64):
    gen = ImageDataGenerator()
    return gen.flow_from_directory(fp, target_size=(224, 224), shuffle=True, class_mode='categorical',
                                   batch_size=batch_size)

def enable_gen_debug(gen, name):
    gen.old_next = gen.next
    def new_next():
        print name
        return gen.old_next()

    gen.next = new_next

def finetune_and_compile_vgg(weights_fp):
    model = build_vgg_model(weights_fp)
    model.pop()
    for l in model.layers:
        l.trainable = False

    model.add(Dense(2, activation='softmax'))
    model.compile(Adam(), 'categorical_crossentropy', metrics=['accuracy'])

    return model



weights_fp = "/home/dpetrovskyi/.keras/models/vgg16.h5"

sample_fp= '/home/dpetrovskyi/fai/sample'
valid_fp = '/home/dpetrovskyi/fai/valid'
to_predict_fp = '/home/dpetrovskyi/fai/to_predict'


model = finetune_and_compile_vgg(weights_fp)

batch_size = 64
train_gen = load_generator(sample_fp, batch_size)
valid_gen = load_generator(valid_fp, batch_size)
to_pred_gen = load_generator(to_predict_fp, batch_size)
to_predict_arr, filenames = get_list_of_images(to_predict_fp)

print 'Fitting'
model.fit_generator(train_gen,
                    steps_per_epoch=train_gen.samples/batch_size,
                    validation_data=valid_gen,
                    validation_steps=valid_gen.samples/batch_size)
print 'Fitting is done'

# res = model.predict_generator(to_pred_gen, (to_pred_gen.samples/8)+1)
res = model.predict(to_predict_arr)
create_submission(res, filenames)

