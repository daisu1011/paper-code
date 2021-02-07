# Core Keras libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Make results reproducible
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

# Other essential libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from numpy import array
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from math import sqrt


# Make our plot a bit formal
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 10}
plt.rc('font', **font)

## prepare data
df1 = pd.read_csv('F:/CLHLS/data/dataset0214.csv')

data1 = df1.values
data1 = data1.astype('float32')

data2 = data1[:,4:]

scaler = MinMaxScaler(feature_range=(0, 1))
data3 = scaler.fit_transform(data2)

n,_ = data3.shape
data4 = []
for i in range(5):
    data4.append(data3[range(i,n,5),:])

data5 = np.array(data4)

trainX = data5[0:3,:,:]
trainY = data5[3,:,:]
trainX = trainX.reshape((-1,3,16))

testX = data5[1:4,:,:]
testX = testX.reshape((-1,3,16))
testY = data5[4,:,:]

## define the lstm model
def lstm_model():
    model = keras.Sequential([
        keras.layers.LSTM(50),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(20, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(16, activation='sigmoid'),
    ])
    model.compile('Adam',
                  loss=keras.losses.MeanSquaredError())
    return model

model = lstm_model()

# Start training
history = model.fit(trainX, trainY, epochs=15, batch_size=64, validation_data=(testX, testY))

# Training curve plot
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Training curve')
    plt.legend([string, 'val_'+string])
    plt.show()

plot_graphs(history, 'loss')

# predict our data
y_predicted = model.predict(testX)

# 'De-normalize' the data
y_predicted_descaled = scaler.inverse_transform(y_predicted)
y_train_descaled = scaler.inverse_transform(trainY)
y_test_descaled = scaler.inverse_transform(testY)

# export result to csv file
result = np.round(y_predicted_descaled)
np.savetxt('./result.csv', result, delimiter=',')