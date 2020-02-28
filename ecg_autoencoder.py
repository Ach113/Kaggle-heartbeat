# -*- coding: utf-8 -*-

import numpy as np
import wfdb, random

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras as keras

import matplotlib.pyplot as plt

ekg_data = wfdb.rdsamp('100', sampto=300000)[0]
ekg_data = ekg_data.transpose()[0]

noisy_data = wfdb.rdsamp('118e00', sampto=300000)[0]
noisy_data = noisy_data.transpose()[0]

samples = 300

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
    
x_noise = []
for i in range(0, noisy_data.shape[0]-samples, samples):
    x_noise.append(noisy_data[i:i+samples])

x_train = np.array(x_train)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_train = keras.utils.normalize(x_train, axis=1)

x_noise = np.array(x_noise)
x_noise = x_noise.reshape(x_noise.shape[0], x_noise.shape[1], 1)
x_noise = keras.utils.normalize(x_noise, axis=1)

train_x, valid_x, train_ground, valid_ground = train_test_split(x_train,
                                                             x_train, 
                                                             test_size=0.2, 
                                                             random_state=13)
data_in = Input(shape=(train_x.shape[1], 1))
model = Model(data_in, autoencoder(data_in))
model.compile(loss='mean_squared_error', optimizer=RMSprop())

model.fit(x_train, x_train, epochs=10)

n = random.randrange(len(x_train))
x = model.predict(x_train)

fig, ax = plt.subplots()
ax.plot(x_train[88], label='real signal')
ax.plot(x[88], label='reconstructed signal')
ax.plot(abs(x[88] - x_train[88]), label='error signal', color='red')
ax.legend()

denoised = model.predict(x_noise)

fig, ax = plt.subplots()
ax.plot(x_noise[88], label='noisy signal')
ax.plot(denoised[88], label='denoised signal')
ax.legend()
