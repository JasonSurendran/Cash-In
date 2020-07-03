### Data Preprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

# Using 70% of Dataset to train
for i in range(60, 1600):
    xtrain.append(training_scaled[i-60:i, 0])
    ytrain.append(training_scaled[i, 0])
xtrain = np.array(xtrain)
ytrain = np.array(ytrain)
xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], 1))

### Recurrent Neural Network
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Run On Test Set

# Calculate RMSE

# Training loss curve

# Graph Results