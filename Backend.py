import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def LTSM(battery, cycle):
    batteries = ['B05', 'B07', 'B18', 'B33', 'B34', 'B46', 'B47', 'B48']
    file_path = './content/' + batteries[battery - 1] + '_discharge_soh.csv'
    dataset = pd.read_csv(file_path)

    X = dataset['cycle'].values
    X = X.reshape(-1, 1)
    y = dataset['SOH'].values

    # Define the number of folds for cross-validation
    n_splits = 5  # You can adjust the number of folds as needed

    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=False)

    # Lists to store predicted values from each fold
    predictions = []

    # Define parameter grid for GridSearchCV
    parameters = {
        'lstm_units': [32, 64, 128],
        'learning_rate': [0.001, 0.01, 0.1],
        'batch_size': [32, 64, 128]
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv=kf)

    # Perform grid search
    grid_search.fit(X_train, y_train)

    # Get best parameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Refit the model with best parameters on the whole training set
    best_model = Sequential()
    best_model.add(LSTM(units=best_params['lstm_units'], input_shape=(1, 1)))
    best_model.add(Dense(1))
    best_model.compile(loss='mean_squared_error', optimizer='adam')

    # Loop over each fold
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = X_train.reshape((X_train.shape[0], 1))
        X_test = X_test.reshape((X_test.shape[0], 1))

        # Train the model
        best_model.fit(X_train, y_train, epochs=50, batch_size=best_params['batch_size'], verbose=2)

        # Make predictions on the test set for this fold
        predictions.append(best_model.predict(X_test))

    # Combine predictions from all folds
    all_predictions = np.concatenate(predictions)

    # Compute mean prediction across all folds
    mean_prediction = np.mean(all_predictions, axis=0)

    # Reshape the cycle input for prediction
    cycle = np.array(cycle).reshape(-1, 1)

    # Make prediction on the input cycle
    predict = best_model.predict(cycle)

    # Evaluation
    mse = mean_squared_error(y_test, mean_prediction)
    mae = mean_absolute_error(y_test, mean_prediction)

    print("Mean squared error:", mse)
    print("Mean absolute error:", mae)

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


