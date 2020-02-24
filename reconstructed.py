# -*- coding: utf-8 -*-

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten

import numpy as np

saved_model = load_model("Saved/cnn.h5")

def get_filter():
    W = saved_model.get_weights()
    return np.array(W[0])

def conv_1d(data, filters, kernel_size):
    assert len(data.shape) == 3, "Expected input to be 3 dimensional"
    N = data.shape[0]
    M = data.shape[1]
    #C = data.shape[2]
    
    channels = list()
    for _ in range(filters):
        filter_ = get_filter()
        matrix = list()
        for n in range(N): # iterate over rows
            row = list()
            for m in range(M): # iterate over columns
                try:
                    conv = [data[n][m], data[n][m+1], data[n][m+2]] * filter_
                    row.append(np.sum(conv))
                except IndexError:
                    pass             
            matrix.append(row)
        channels.append(matrix)
    return np.transpose(np.asarray(channels), (1, 2, 0))

def max_pooling1d(layer, pool_size, strides, padding='same'):
    assert len(layer.shape) == 3, "Expected input to be 3 dimensional"
    N = layer.shape[0]
    M = layer.shape[1]
    C = layer.shape[2]
    
    channels = list()
    for c in range(C): # iterate over channels
        matrix = list()
        for n in range(N): # iterate over rows
            row = list()
            for m in range(M): # iterate over columns
                try:
                    val = max(layer[n][m][c], layer[n][m+1][c])
                    row.append(val)
                except IndexError:
                    pass               
            matrix.append(row)
        channels.append(matrix)
            
    return np.transpose(np.array(channels), (1,2,0))

def flatten(layer):
    assert len(layer.shape) == 3, "Expected input to be 3 dimensional"
    batch = layer.shape[0]
    channels = layer.shape[2]
    
    matrix = list()
    for b in range(batch):
        row = list()
        for c in range(channels):
            if c==0:
                row = layer[b,:,c]
            else:
                row = np.concatenate((row, layer[b,:,c]), axis=0)
        matrix.append(row)
    return np.array(matrix)

x = np.random.random((100, 187, 1))

k1 = Conv1D(filters=10, kernel_size=3, kernel_initializer='ones')(x)
k2 = MaxPooling1D(pool_size=2, strides=1, padding='valid')(k1)
k3 = Flatten()(k2)

x1 = conv_1d(x, filters=10, kernel_size=3)
x2 = max_pooling1d(x1, pool_size=2, strides=1, padding='same')
x3 = flatten(x2)

print(k3.shape)
print(x3.shape)


