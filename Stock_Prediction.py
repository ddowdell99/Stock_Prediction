# Description: This program uses an artificial recurrent neural network called Long Short Term Memory (LSTM) to predict the closing stock price
#              of a corporation using the past 60 day stock price. 

# Importing Libraries

import math
from pandas_datareader import data as pdr
import pandas_datareader.data as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import yfinance as yf
yf.pdr_override()
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Grab the stock quote
df = pdr.get_data_yahoo('AAPL', start='2012-01-01', end='2023-01-16')

# Show Data
# print(df)

# Grab number of rows and columns in data set
# (rows, columns)
# print(df.shape)

# Visualize closing price history

plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
# plt.show()

# Filter dataframe to only show closing price history

data = df.filter(['Close'])

# Convert the dataframe to a numpy array (NUMPY IS MORE EFFICIENT)
dataset = data.values

# Grab number of rows to train the model on
training_data_length = math.ceil(len(dataset) * .8) # Going to train on 80% of the data *** can be adjusted.

# Scale the data (Turning all the data to numbers between 0 and 1, helps with the modeling and scaling of data)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Create the training data set
# Create the scaled training data set

training_data = scaled_data[0:training_data_length, :] 
# print(training_data)


x_train = []
y_train = []

for i in range(60, len(training_data)):
    x_train.append(training_data[i-60:i,0])
    y_train.append(training_data[i, 0])
    # testing what this is doing

    # if i <=60:
    #     print(x_train)
    #     print(y_train)

# Converting x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1)) # Numpy reshape can be used to turn 2d data tables into 3d

# print(x_train.shape)

# Building the LSTM model

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model

model.compile(optimizer='adam', loss='mean_squared_error')

# Train (fit) the model

model.fit(x_train, y_train, batch_size=1, epochs=1)

# Create testing data set
# Creating new array containing scaled values from index 2162 to  2500 (testing data)
testing_data = scaled_data[training_data_length - 60: , :]

# Create the data sets x_test and y_test

x_test = []
y_test = dataset[training_data_length:, :]

for i in range(60, len(testing_data)):
    x_test.append(testing_data[i-60:i,0])

# Convert the data to numpy array
x_test = np.array(x_test)

# Reshape the data

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

# Get the models predicted price values (reversing/inversing the scaled data back to actual usable values)
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
print(rmse)