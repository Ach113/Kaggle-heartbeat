# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow as tf
import os

from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.layers import Dense, Conv1D, Flatten, Masking, MaxPooling1D, Dropout, Input
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

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

def evaluate_model(model, x_test, y_test):
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    y_pred = model.predict(x_test)
    y_pred = [np.argmax(x) for x in y_pred]
    y_true = [np.argmax(x) for x in y_test]
    plot_confusion_matrix(y_true, y_pred, acc)
    print(classification_report(y_true, y_pred))
    
def plot_samples(X, y):
    classes = set(y)
    labels = ['Non-ecotic beats (normal beat)', 'Supraventricular ectopic beats', 'Ventricular ectopic beats',
              'Fusion Beats', 'Unknown Beats']
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in classes:
        x = np.argwhere(y==i)[0][0]
        ax.plot(X[x], label=labels[i])
        ax.legend()
        
def define_CNN():
    input_layer = Input(shape=(X_train.shape[1], 1))
    x = Conv1D(64, kernel_size=3, activation='relu')(input_layer)
    x = Conv1D(32, kernel_size=3, activation='relu')(x)
    
    x = MaxPooling1D(pool_size=2, strides=1, padding='same')(x)
    x = Flatten()(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    output = Dense(5, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output)
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    
    return model
        
train = pd.read_csv("mitbih_train.csv")
test = pd.read_csv("mitbih_test.csv")

# last last column contains target values, 5 classes in total

X_train = train.iloc[:, :-1].values
y = train.iloc[:, -1].values.astype(int)

X_train = tf.keras.utils.normalize(X_train, axis=1)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
y_train = tf.keras.utils.to_categorical(y)

plot_samples(X_train, y)

if not os.path.exists("Saved/cnn.h5"):
    model = define_CNN()
    model.fit(X_train, y_train, epochs=5, batch_size=200)
    model.save("Saved/cnn.h5")
else:
    model = load_model("Saved/cnn.h5")

# validate on test data

X_test = test.iloc[:, :-1].values
y = test.iloc[:, -1].values

X_test = tf.keras.utils.normalize(X_test, axis=1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
y_test = tf.keras.utils.to_categorical(y)

evaluate_model(model, X_test, y_test)





