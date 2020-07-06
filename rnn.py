####### Data Preprocessing ####################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Importing Data
dataset_train = pd.read_csv('Manulife_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Scale features between 0-1
scaling = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = scaling.fit_transform(training_set)

# Data structure
# Remember the last 2 months
xtrain = []
ytrain = []

# Using ~70% of Dataset to train (1600/2300)
for i in range(60, 1600):
    xtrain.append(training_set_scaled[i-60:i, 0])
    ytrain.append(training_set_scaled[i, 0])
xtrain, ytrain = np.array(xtrain), np.array(ytrain)

xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], 1))
################################################################################



####### Recurrent Neural Network ###############################################
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential

# Starting RNN
rnn = Sequential()

# First LSTM layer w/ dropout 
rnn.add(LSTM(units = 50, return_sequences = True, input_shape = (xtrain.shape[1], 1)))
rnn.add(Dropout(0.2))

# Second LSTM layer w/ dropout 
rnn.add(LSTM(units = 50, return_sequences = True))
rnn.add(Dropout(0.2))

# Third LSTM layer w/ dropout 
rnn.add(LSTM(units = 50))
rnn.add(Dropout(0.2))

# Output layer
rnn.add(Dense(units = 1))

# Compiling the RNN
# Adam optimization is a stochastic gradient descent method
# It is based on adaptive estimation of first-order and second-order moments.
rnn.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Train the model by fitting the RNN for the training set
history = rnn.fit(xtrain, ytrain, epochs = 100, batch_size = 32)
####################################################################################



####### Run On Test Set ############################################################
# Getting real stock price from csv
dataset_test = pd.read_csv('Manulife_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting prediction stock price
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = scaling.transform(inputs)
xtest = []
for i in range(60, 760):
    xtest.append(inputs[i-60:i, 0])
xtest = np.array(xtest)
xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 1))
predicted_stock_price = rnn.predict(xtest)
predicted_stock_price = scaling.inverse_transform(predicted_stock_price)
####################################################################################



####### Calculate RMSE #############################################################
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
print(rmse)
####################################################################################



####### Training Loss Curve ########################################################
loss = history.history['loss']
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, 'ro', label='Training Loss')
plt.legend()
plt.show()
####################################################################################



####### Graph Results ##############################################################
plt.plot(real_stock_price, color = 'red', label = 'Real Stock Price')
plt.plot(predicted_stock_price, color = 'black', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Days Forward')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
####################################################################################

