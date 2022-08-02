#!/bin/env python3

from pickletools import optimize
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def ANN(x_train, y_train, x_predict, BATCH_SIZE, lr, EPOCHS = 100):
    # the node of hidden
    n_hidden_1 = 50
    n_hidden_2 = 50
    num_input = len(x_train[0])
    num_output = len(y_train[0])
    # create model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(n_hidden_1, input_dim = num_input, activation='relu'))
    model.add(tf.keras.layers.Dense(n_hidden_2, activation='relu'))
    model.add(tf.keras.layers.Dense(num_output, activation='relu'))
    model.summary()
    
    # set loss function
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse', metrics=['acc'])

    # train model
    history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # evaluate the model
    _, accuracy = model.evaluate(x_train, y_train)
    print('Accuracy of ANN model:%.2f %%' % (accuracy*100))

    # predictions with the model
    predictions = model.predict(x_predict)
    #for i in range(len(x_train)):
    #    print('%s => %s (expected %s)' % (x_train[i].tolist(), predictions[i], y_train[i]))
    return predictions

#print(x_train)
dataset = np.loadtxt('train.dat', delimiter=',')

new = []
for info in range(len(dataset)):
    if dataset[info][0] < 1000000:
        new.append(dataset[info])
new = np.array(new)

x_train = new[:400,:2]/np.array([1000,1])
print(x_train)
y_train = new[:400,2:4]

BATCH_SIZE = 32
EPOCHS = 2000
lr = 0.001
x_predict = np.array([[0.974,0.27]])

predictions = ANN(x_train, y_train, x_predict, BATCH_SIZE,lr, EPOCHS)
print(predictions)


