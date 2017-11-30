from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x = np.random.uniform(0,1, 1000)
y = [int(a>0.5) for a in x]
x_val, y_val = x,y


# 2-class logistic regression in Keras
model = Sequential()
model.add(Dense(1, activation='sigmoid', input_shape=(1,)))
model.compile(optimizer='rmsprop', loss='binary_crossentropy')
model.fit(x, y, nb_epoch=10, validation_data=(x_val, y_val))

