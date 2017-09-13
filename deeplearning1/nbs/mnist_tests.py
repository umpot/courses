from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers.core import Lambda, Dense, Flatten, Dropout, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam

import matplotlib.pyplot as plt

def plot_accuracy(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def plot_loss(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def one_hot(train, test):
    encoder = OneHotEncoder()
    train = encoder.fit_transform(train.reshape(len(train), 1))
    test = encoder.transform(test.reshape(len(test), 1))

    return train, test

(train_arr, train_labels), (test_arr, test_labels) = mnist.load_data()
train_target, test_target = one_hot(train_labels, test_labels)
train_target, test_target= train_target.toarray(), test_target.toarray()

dt_mean = train_arr.mean()
dt_std = train_arr.std()

def norm_input(x):
    return (x-dt_mean)/dt_std

def create_linear():
    model = Sequential()
    model.add(Lambda(norm_input, output_shape=(28, 28), input_shape=(28, 28)))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_arr, train_target, epochs=1, validation_data=(test_arr, test_target))

    return model


def create_mlp_two_layers():
    model = Sequential()
    model.add(Lambda(norm_input, output_shape=(28, 28), input_shape=(28, 28)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    h = model.fit(train_arr, train_target, epochs=5, validation_data=(test_arr, test_target), verbose=2)

    return model

def create_mlp_tree_layers():
    model = Sequential()
    model.add(Lambda(norm_input, output_shape=(28, 28), input_shape=(28, 28)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    h = model.fit(train_arr, train_target, epochs=10, validation_data=(test_arr, test_target), verbose=2)

    return model


model = Sequential()
model.add(Lambda(norm_input, output_shape=(28, 28), input_shape=(28, 28)))
model.add(Reshape((28,28,1)))
model.add(Conv2D(128, (3,3)))
model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))

model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

h = model.fit(train_arr, train_target, epochs=10, validation_data=(test_arr, test_target), verbose=2)




