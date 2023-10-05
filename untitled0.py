# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 19:53:43 2023

@author: Chenhao Sun
"""

#%% Preparation
import pandas as pd

df = pd.read_excel(r'D:\Users\Chenhao Sun\Desktop\Data (Shared).xlsx')
print(df.head())

#%% Linear Regression Model

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Split the data into features (X) and the target variable (y)
X = df[['Prices']]
y = df['YTM (in %)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Create a Linear Regression model and train
model = LinearRegression()
model.fit(X_train, y_train)

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R-squared: {r2}')

#%% Random Forest Regressor Model (Enter parameters manually)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Split the data into training and testing sets
X = df[['Prices']]
y = df['YTM (in %)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.75, random_state = 42)

# Create and train the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators = 270, random_state = 42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R-squared: {r2}')

#%% Random Forest Regressor Model (test best combination of n_estimators and test_size)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Define the parameter grid for n_estimators
n_estimators_values = list(range(100, 501, 10))

# Create the Random Forest Regressor model
rf_model = RandomForestRegressor(random_state = 42)

# Split the data into training and testing sets
X = df[['Prices']]
y = df['YTM (in %)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Initialize variables to store the best hyperparameters
best_mse = float('inf')
best_hyperparameters = {'n_estimators': None, 'test_size': None}

# Perform nested grid search for both hyperparameters
for n_estimators in n_estimators_values:
    for test_size in [round(0.10 + 0.05 * i, 2) for i in range(17)]:
        # Split the data into training and testing sets for the current test_size
        X_train_inner, X_test_inner, y_train_inner, y_test_inner = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Create and fit the Random Forest Regressor model with the current n_estimators
        rf_model.set_params(n_estimators=n_estimators)
        rf_model.fit(X_train_inner, y_train_inner)
        
        # Make predictions with the model
        y_pred = rf_model.predict(X_test_inner)
        
        # Calculate the mean squared error for this combination
        mse = mean_squared_error(y_test_inner, y_pred)
        
        # Check if the current combination resulted in a lower MSE
        if mse < best_mse:
            best_mse = mse
            best_hyperparameters = {'n_estimators': n_estimators, 'test_size': test_size}

# Print the best hyperparameters
print("Best Hyperparameters:", best_hyperparameters)

# Train the final model with the best hyperparameters
best_model = RandomForestRegressor(n_estimators=best_hyperparameters['n_estimators'], random_state = 42)
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X, y, test_size=best_hyperparameters['test_size'], random_state = 42)
best_model.fit(X_train_final, y_train_final)

# Make predictions with the best model
y_pred = best_model.predict(X_test_final)

# Evaluate the best model
mse = mean_squared_error(y_test_final, y_pred)
print("Mean Squared Error with Best Model:", mse)

#%% Useless Part

# Extract relevant columns from the DataFrame
date_column = 'Date'
price_column = 'Prices'
ytm_column = 'T 0⅛ 10/31/22 Govt - Mid Yield To Maturity (Raw data)'

# Create the bond_data list from the DataFrame
bond_data = [(price, ytm) for price, ytm in zip(df[price_column], df[ytm_column])]

print(bond_data)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

def validate_and_calculate_error(bond_data):
    actual_ytm = [ytm for _, ytm in bond_data]

    # Calculate error metrics
    mse = mean_squared_error(actual_ytm, [0] * len(actual_ytm))
    mae = mean_absolute_error(actual_ytm, [0] * len(actual_ytm))
    r_squared = r2_score(actual_ytm, [0] * len(actual_ytm))
    rmse = math.sqrt(mse)

    return {
        'MSE': mse,
        'MAE': mae,
        'R-squared': r_squared,
        'RMSE': rmse
    }

error_metrics = validate_and_calculate_error(bond_data)
print("Error Metrics:")
for metric, value in error_metrics.items():
    print(f"{metric}: {value}")

#%%
import scipy.optimize as so
import numpy as np
import pandas as pd
from datetime import datetime

df = pd.read_excel(r'D:\Users\Chenhao Sun\Desktop\Data (Shared).xlsx')

def YTM(PV, C, k, d, n, M, TS):
 def f(y):
  coupon=[]
  for i in np.arange(0, n):
            coupon.append((C / k) / pow(1 + y / k, d / TS + i))
  return np.sum(coupon) + M / pow(1 + y / k, d / TS + n - 1) - PV
 return so.fsolve(f, 0)

k = 2
n = 4
M = 100
C = 0.125 

coupon_dates = ['04/30/2021', '10/31/2021', '04/30/2022', '10/31/2022']
issue_date = datetime(2020, 10, 31)

while True:
    D_str = input('Date (MM/DD/YY): ')
    try:
        D = datetime.strptime(D_str, '%m/%d/%Y')
        issue_date = datetime(2020, 10, 31)
        maturity_date = datetime(2022, 10, 31)
        if D < issue_date:
            print('The entered date is earlier than the bond issue date (10/31/2020). Please enter a valid date.')
        elif D > maturity_date:
            print('The entered date is later than the bond maturity date (10/31/2022). Please enter a valid date.')
        else:
            break
    except ValueError:
        print('The date format does not match MM/DD/YYYY format. Please enter a valid date.')
        
if D <= issue_date:
    TS = (datetime(D.year, 4, 30).replace(year = D.year) - issue_date).days
    d = (datetime(D.year, 10, 31).replace(year = D.year) - D).days
else:
    for i, coupon_date_str in enumerate(coupon_dates):
        coupon_date = datetime.strptime(coupon_date_str, '%m/%d/%Y')
        if D < coupon_date:
            TS = (coupon_date - issue_date).days
            d = (coupon_date - D).days
            break

PV = df[df['Date'] == D]['Prices'].values[0]

Bond_yield=YTM(PV, C, k, d, n, M, TS)[0]
estimate_ytm = np.round(Bond_yield, 10)
print(estimate_ytm)

actual_ytm = df[df['Date'] == D]['YTM (in %)'].values[0]
print(estimate_ytm - actual_ytm)














