import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def LTSM(battery, cycle):
    batteries = ['B05', 'B07', 'B18', 'B33', 'B34', 'B46', 'B47', 'B48']
    file_path = './content/' + batteries[battery - 1] + '_discharge_soh.csv'
    dataset = pd.read_csv(file_path)

    # Restructure features X and target y
    X = dataset['cycle'].values.reshape(-1, 1)  # Reshaping to (-1, 1) for one feature
    y = dataset['SOH'].values

    # Normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Reshape data for LSTM input (samples, time steps, features)
    # Assuming you have 'n' timesteps for each sample
    n_steps = 10  # Adjust this value based on your data
    n_features = 1  # Since you have one feature after reshaping

    # Define LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))  # Output layer

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print("Test Loss:", loss)

    # Predict SOH for test data
    y_pred = model.predict(X_test)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Absolute Error:", mae)

    # Calculate R-squared Score
    r2 = r2_score(y_test, y_pred)
    print("R-squared Score:", r2)

    cycle = np.array(cycle).reshape(-1, 1)
    predict = model.predict(cycle)

    return predict.tolist()


 def linear(battery, cycle):
    batteries = ['B05', 'B07', 'B18', 'B33', 'B34', 'B46', 'B47', 'B48']
    file_path = './content/' + batteries[battery - 1] + '_discharge_soh.csv'
    dataset = pd.read_csv(file_path)

    X = dataset['cycle'].values  
    X = X.reshape(-1, 1)
    y = dataset['SOH'].values     

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()

    # Define parameter grid for GridSearchCV
    parameters = {'fit_intercept': [True, False]}

    # Initialize GridSearchCV
    grid_search = GridSearchCV(model, parameters, cv=5)

    # Perform grid search
    grid_search.fit(X_train, y_train)

    # Get best parameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Refit the model with best parameters on the whole training set
    best_model = LinearRegression(**best_params)
    best_model.fit(X_train, y_train)
    
    # Make prediction on the input cycle
    cycle = np.array(cycle).reshape(-1, 1)
    predict = best_model.predict(cycle)

    # Evaluate on test set
    test_predictions = best_model.predict(X_test)
    mse_test = mean_squared_error(y_test, test_predictions)
    mae_test = mean_absolute_error(y_test, test_predictions)
    r2 = r2_score(y_test, test_predictions)
     
    print("R-squared Score:", r2)
    print("Best Parameters:", best_params)
    print("Best Score (CV MSE):", best_score)
    print("Mean Squared Error (Test):", mse_test)
    print("Mean Absolute Error (Test):", mae_test)

    return predict.tolist()

  
def predict(model,battery,cycle):
  if model=='LTSM':
    return LTSM(battery,cycle)
  else:
    return linear(battery,cycle)


