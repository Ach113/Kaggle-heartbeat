# -*- coding: utf-8 -*-

import numpy as np
import wfdb, random

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

def plot_results(true_signal, reconstructed_signal, n=None):
    fig, axs = plt.subplots(nrows=3, figsize=(6, 4))
    for ax in axs:
        if n is None:
            n = random.randrange(len(x_train))
        ax.plot(true_signal[n], label='real')
        ax.plot(reconstructed_signal[n], label='reconstructed')
        ax.plot(reconstructed_signal[n] - true_signal[n], label='error', color='red')
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower left')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle('Comparison of real and reconstructed signals')


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

data_in = Input(shape=(x_train.shape[1], 1))
model = Model(data_in, autoencoder(data_in))
model.compile(loss='mean_squared_error', optimizer=RMSprop())

model.fit(x_train, x_train, epochs=10)

x = model.predict(x_train)

plot_results(100*x_train, 100*x)
