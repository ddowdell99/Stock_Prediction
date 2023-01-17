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
print(df)

# Grab number of rows and columns in data set
# (rows, columns)
print(df.shape)

# Visualize closing price history

plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()

# Filter dataframe to only show closing price history

data = df.filter(['Close'])

# Convert the dataframe to a numpy array
dataset = data.values

# Grab number of rows to train the model on
training_data_length = math.ceil(len(dataset) * .8) # Going to train on 80% of the data *** can be adjusted.

# Scale the data (Turning all the data to numbers between 0 and 1, helps with the modeling and scaling of data)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Create the training data set
# Create the scaled training data set

training_data = scaled_data[0:training_data_length] 
print(training_data)

x_train = []
y_train = []