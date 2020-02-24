# -*- coding: utf-8 -*-

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import tensorflow.keras as keras

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, accuracy):
    df = pd.DataFrame(data=confusion_matrix(y_true, y_pred),
                      columns = ['N', 'S', 'V', 'F', 'Q'],
                      index = ['N', 'S', 'V', 'F', 'Q'])
    plt.figure(figsize=(5,4))
    sns.heatmap(df, annot=True, fmt='g')
    plt.title('CNN classifier \nAccuracy: {0:.3f}'.format(accuracy))
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def evaluate_model(y_pred, y_true):
    acc = accuracy_score(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, acc)
    print(classification_report(y_true, y_pred))

saved_model = load_model("Saved/cnn.h5")
test = pd.read_csv("mitbih_test.csv")

def relu(x):
    return x * (x > 0)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_filter():
    W = saved_model.get_weights()
    return np.array(W[0])

def conv_1d(data, filters, kernel_size, activation='relu'):
    assert len(data.shape) == 3, "Expected input to be 3 dimensional"
    N = data.shape[0]
    M = data.shape[1]
    
    channels = list()
    for f in range(filters):
        print("> filter no. ", f+1)
        filter_ = get_filter()
        matrix = list()
        for n in range(N): # iterate over rows
            row = list()
            for m in range(M): # iterate over columns
                try:
                    conv = np.matmul(np.array([data[n][m], data[n][m+1], data[n][m+2]]), filter_)
                    row.append(np.sum(conv))
                except IndexError:
                    pass             
            matrix.append(row)
        channels.append(matrix)
    M = np.transpose(np.asarray(channels), (1, 2, 0))
    print("> Convolution complete")
    
    if activation=='relu':
        return relu(M)
    
    return M

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
    print("> Max pooling complete")
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
    print("> Flattening complete")
    return np.array(matrix)

def dense1(layer, activation='relu'):
    W = saved_model.get_weights()[1]
    b = saved_model.get_weights()[2]
    
    x = np.matmul(layer, W) + b
    print("> Dense1 complete")
    
    if activation=='relu':
        return relu(x)
    
    return x

def dense2(layer, activation='relu'):
    W = saved_model.get_weights()[3]
    b = saved_model.get_weights()[4]
    
    x = np.matmul(layer, W) + b
    print("> Dense2 complete")
    
    if activation=='relu':
        return relu(x)
    
    return x
    
def dense3(layer, activation='relu'):
    W = saved_model.get_weights()[5]
    b = saved_model.get_weights()[6]
    
    x = np.matmul(layer, W) + b
    print("> Dense3 complete")
    
    if activation=='relu':
        return relu(x)
    if activation=='softmax':
        return softmax(x)
    
    return x

def reconstructed_model(Input):
    x = conv_1d(Input, filters=10, kernel_size=3)
    x = max_pooling1d(x, pool_size=2, strides=1, padding='same')
    x = flatten(x)
    x = dense1(x)
    x = dense2(x)
    x = dense3(x, activation='softmax')
    
    print("> Prediction complete")
    return x

# evaluate model

X_test = test.iloc[:, :-1].values
y = test.iloc[:, -1].values

X_test = keras.utils.normalize(X_test, axis=1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
y_test = keras.utils.to_categorical(y)

print("> starting prediction")
print(f"> Validation data shape: {X_test.shape}")
prediction = reconstructed_model(X_test)

y_pred = [np.argmax(x) for x in prediction]
y_dense = [np.argmax(x) for x in y_test]

evaluate_model(y_pred, y_dense)

# keras model
prediction = saved_model.predict(X_test)

y_pred = [np.argmax(x) for x in prediction]
y_dense = [np.argmax(x) for x in y_test]

evaluate_model(y_pred, y_dense)
