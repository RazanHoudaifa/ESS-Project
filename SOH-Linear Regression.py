from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error


file_path = './content/B05_discharge_soh_2.csv'
dataset = pd.read_csv(file_path)

# Preparing the dataset for Linear Regression
#We select only the cycle feature
X = dataset['cycle'].values  
X = X.reshape(-1, 1)
y = dataset['SOH'].values     

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
yhat_train = model.predict(X_train)
yhat_test = model.predict(X_test)

# Calculate RMSE and MAE for test set

rmse_test = math.sqrt(mean_squared_error(y_test, yhat_test))
mae_test = mean_absolute_error(y_test, yhat_test)
print('Test RMSE: %.3f' % rmse_test)
print('Test MAE: %.3f' % mae_test)

# Plotting predictions
sns.set_style("darkgrid")
plt.figure(figsize=(12, 8))
cycle = dataset['cycle'].values
# Plot training data
plt.plot(cycle[:len(y_train)], y_train, label='Real Data (Train)', linewidth=3, color='r')
# Plot testing data
plt.plot(cycle[len(y_train):], y_test, label='Real Data (Test)', linewidth=3, color='b')
# Plot training predictions
plt.plot(cycle[:len(yhat_train)], yhat_train, label='Linear Regression Prediction (Train)', linestyle='--', color='g')
# Plot testing predictions
plt.plot(cycle[len(y_train):], yhat_test, label='Linear Regression Prediction (Test)', linestyle='--', color='m')
#just tidying up the plot
plt.legend(prop={'size': 16})
plt.ylabel('SoH', fontsize=15)
plt.xlabel('Cycle', fontsize=15)
plt.title("SOH Prediction with Linear Regression", fontsize=15)
plt.tight_layout()
plt.show()
