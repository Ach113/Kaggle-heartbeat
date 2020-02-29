# -*- coding: utf-8 -*-

from tensorflow.keras.models import load_model
import tensorflow.keras as keras

from sklearn.metrics import classification_report, accuracy_score

import numpy as np
import pandas as pd

def relu(x):
    return x * (x > 0)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def conv_1d(data, filters, kernel_size, filt, padding='valid', activation='relu'):
    assert len(data.shape) == 3, "Expected input to be 3 dimensional"
    N = data.shape[0]
    M = data.shape[1]
    C = data.shape[2]
    
    channels = list()
    for f in range(filters):
        filter_ = filt
        matrix = list()
        for n in range(N): # iterate over rows 
            row = list()      
            for m in range(M - kernel_size + 1): # iterate over columns
                conv, pad_start, pad_end = 0, 0, 0
                for c in range(C):
                    if n == 0 and padding=='same':
                        pad_start += data[n][m+1][c]*filter_[1][c][f] + data[n][m+2][c]*filter_[2][c][f]
                    elif (n == N - 1) and padding=='same':
                        pad_end += data[n][m][c]*filter_[0][c][f] + data[n][m+1][c]*filter_[1][c][f]
                    conv += (data[n][m][c]*filter_[0][c][f]) + (data[n][m+1][c]*filter_[1][c][f]) + (data[n][m+2][c]*filter_[2][c][f])
                if m == 0 and padding=='same':
                    row.append(pad_start)
                row.append(conv)
                if (m == M - kernel_size) and padding=='same':
                    row.append(pad_end)
            matrix.append(row)
        channels.append(matrix)
    M = np.transpose(np.asarray(channels), (1, 2, 0))
    print("> Convolution complete")
    
    return relu(M)

def max_pooling1d(layer, pool_size, strides, padding='valid'):
    assert len(layer.shape) == 3, "Expected input to be 3 dimensional"
    N = layer.shape[0]
    M = layer.shape[1]
    C = layer.shape[2]
    
    channels = list()
    for c in range(C): # iterate over channels
        matrix = list()
        for n in range(N): # iterate over rows
            row = list()
            for m in range(M - pool_size + 1): # iterate over columns
                val = max(layer[n][m][c], layer[n][m+1][c])
                row.append(val)  
                if (padding == 'same') and (m == M - pool_size):
                    row.append(layer[n][m][c])
            matrix.append(row)
        channels.append(matrix)
    print("> Max pooling complete")
    return np.transpose(np.array(channels), (1,2,0))

def flatten(layer):
    assert len(layer.shape) == 3, "Expected input to be 3 dimensional"
    batch = layer.shape[0]
    columns = layer.shape[1]
    channels = layer.shape[2]
    
    matrix=list()
    for b in range(batch):
        row = list()
        for i in range(columns):
            for c in range(channels):
                row.append(layer[b,i,c])
        matrix.append(row)
    print("> Flattening complete")
    return np.array(matrix)

def dense1(layer):
    W = saved_model.get_weights()[2]
    b = saved_model.get_weights()[3]
    
    x = np.matmul(layer, W) + b
    print("> Dense 1 complete")

    return relu(x)

def dense2(layer):
    W = saved_model.get_weights()[4]
    b = saved_model.get_weights()[5]
    
    x = np.matmul(layer, W) + b
    print("> Dense 2 complete")  
    
    return relu(x)
    
def dense3(layer, activation='softmax'):
    W = saved_model.get_weights()[6]
    b = saved_model.get_weights()[7]
    
    x = np.matmul(layer, W) + b
    print("> Dense 3 complete")
    
    return softmax(x)

def reconstructed_model(Input):
    x = conv_1d(Input, filters=10, kernel_size=3, filt=saved_model.get_weights()[0], padding='same')
    x = conv_1d(x, filters=5, kernel_size=3, filt=saved_model.get_weights()[1], padding='same')
    x = max_pooling1d(x, pool_size=2, strides=1, padding='same')
    x = flatten(x)
    x = dense1(x)
    x = dense2(x)
    x = dense3(x)
    
    return x

# evaluate model

saved_model = load_model("Saved/cnn.h5")
test = pd.read_csv("Data/mitbih_test.csv")

X_test = test.iloc[:, :-1].values
y = test.iloc[:, -1].values

X_test = keras.utils.normalize(X_test, axis=1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
y_test = keras.utils.to_categorical(y)

print("> starting prediction")
prediction = reconstructed_model(X_test)
print("> Prediction complete")

y_pred = [np.argmax(x) for x in prediction]
y_dense = [np.argmax(x) for x in y_test]

score = accuracy_score(y_dense, y_pred) # 0.956 accuracy, same as Keras model
print(classification_report(y_dense, y_pred))
