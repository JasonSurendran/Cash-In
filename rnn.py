####### Data Preprocessing ####################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Importing Data
dataset = pd.read_csv('Manulife_Stock_Price_Train.csv')
training = dataset.iloc[:, 1:2].values

# Scale features between 0-1
scaling = MinMaxScaler(feature_range = (0, 1))
training_scaled = scaling.fit_transform(training)

# Data structure
# Remember the last 2 months
xtrain = []
ytrain = []

# Using ~70% of Dataset to train (1600/2300)
for i in range(60, 1600):
    xtrain.append(training_scaled[i-60:i, 0])
    ytrain.append(training_scaled[i, 0])
xtrain = np.array(xtrain)
ytrain = np.array(ytrain)
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
rnn.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
rnn.add(Dropout(0.2))

# Second LSTM layer w/ dropout 
rnn.add(LSTM(units = 50, return_sequences = True))
rnn.add(Dropout(0.2))

# Third LSTM layer w/ dropout 
rnn.add(LSTM(units = 50, return_sequences = True))
rnn.add(Dropout(0.2))

# Fourth LSTM layer w/ dropout 
rnn.add(LSTM(units = 50, return_sequences = True))
rnn.add(Dropout(0.2))

# Fifth LSTM layer w/ dropout 
rnn.add(LSTM(units = 50))
rnn.add(Dropout(0.2))

# Output layer
rnn.add(Dense(units = 1))

# Compiling the RNN
# Adam optimization is a stochastic gradient descent method
# It is based on adaptive estimation of first-order and second-order moments.
rnn.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Train the model by fitting the RNN for the training set
history = rnn.fit(X_train, y_train, epochs = 100, batch_size = 32)
####################################################################################



####### Run On Test Set ############################################################

####################################################################################



####### Calculate RMSE #############################################################

####################################################################################



####### Training Loss Curve ########################################################
loss = history.history['loss']
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, 'ro', label='Training Loss')
plt.legend()
plt.show()
####################################################################################



####### Graph Results ##############################################################

####################################################################################