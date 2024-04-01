import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

def LTSM(battery,cycle):
 batteries=['B05', 'B07', 'B18', 'B33', 'B34', 'B46', 'B47', 'B48']
 file_path = './content/'+batteries[battery-1] +'_discharge_soh.csv' 
 dataset = pd.read_csv(file_path)


 features = ['terminal_voltage', 'terminal_current', 'temperature', 'charge_current', 'charge_voltage', 'cycle']
 X = dataset['cycle'].values
 X.reshape(-1, 1)
 y = dataset['SOH'].values


 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


 X_train = X_train.reshape((X_train.shape[0], 1))
 X_test = X_test.reshape((X_test.shape[0], 1))

 # Define the LSTM model
 model = Sequential()
 model.add(LSTM(64, input_shape=(1, 1)))
 model.add(Dense(1))
 model.compile(loss='mean_squared_error', optimizer='adam')

 # Train the model
 history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), verbose=2)
 print(cycle)
 print(type(cycle))
 cycle=np.array(cycle).reshape(-1,1)
 predict=model.predict(cycle)
 return predict.tolist()


def linear(battery,cycle):
  batteries=['B05', 'B07', 'B18', 'B33', 'B34', 'B46', 'B47', 'B48']
  file_path = './content/'+batteries[battery-1]+'_discharge_soh.csv'
  dataset = pd.read_csv(file_path)

  X = dataset['cycle'].values  
  X = X.reshape(-1, 1)
  y = dataset['SOH'].values     
 
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

  model = LinearRegression()
  model.fit(X_train, y_train)
  cycle=np.array(cycle).reshape(-1,1)
  predict=model.predict(cycle)
  return predict.tolist()

  
def predict(model,battery,cycle):
  if model=='LTSM':
    return LTSM(battery,cycle)
  else:
    return linear(battery,cycle)


