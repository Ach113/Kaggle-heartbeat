# -*- coding: utf-8 -*-

import numpy as np
import struct

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras as keras

import matplotlib.pyplot as plt

def read_ekg_data(input_file):
    """
    Read the EKG data from the given file.
    """
    with open(input_file, 'rb') as input_file:
        data_raw = input_file.read()
    n_bytes = len(data_raw)
    n_shorts = n_bytes/2

    unpack_string = '<%dh' % n_shorts
    
    data_shorts = np.array(struct.unpack(unpack_string, data_raw)).astype(float)
    return data_shorts

ekg_data = read_ekg_data('a02.dat')
print(ekg_data.shape) # (3182000,)
samples = 500

def autoencoder(data_in):
    # Encoder:
    x = Conv1D(128, kernel_size=3, padding='same', activation='relu')(data_in)
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = Conv1D(32, kernel_size=3, padding='same', activation='relu')(x)
    # Decoder:
    x = Conv1D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = UpSampling1D(2)(x)
    data_out = Conv1D(1, kernel_size=3, padding='same', activation='tanh')(x)
    
    return data_out

x_train = []
for i in range(0, ekg_data.shape[0]-samples, samples):
    x_train.append(ekg_data[i:i+samples])

x_train = np.array(x_train)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_train = keras.utils.normalize(x_train, axis=1)

train_x, valid_x, train_ground, valid_ground = train_test_split(x_train,
                                                             x_train, 
                                                             test_size=0.2, 
                                                             random_state=13)
data_in = Input(shape=(train_x.shape[1], 1))
model = Model(data_in, autoencoder(data_in))
model.compile(loss='mean_squared_error', optimizer=RMSprop())

model.fit(train_x, train_ground, validation_data=(valid_x, valid_ground), epochs=5)

x = model.predict(x_train)

fig, ax = plt.subplots()
ax.plot(x_train[88], label='real signal')
ax.plot(x[88], label='reconstructed signal')
ax.legend()

