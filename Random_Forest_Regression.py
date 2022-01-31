# RANDOM FOREST REGRESSION

# Ensemble learning = Using multiple algorithms/same algorithm multiple times, and put them together to a much more powerful algorithm

# Pick ar random K data points from the training set
# Build the decision tree associated to these K data points
# Choose the number Ntree of trees you want to build and repeat steps 1 anf 2
# For a new data point, Make each one of your Ntree trees predict the value of Y to for the data point in question, and assign the newdata point the average across all of the predicted Y values.


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1) Data preprocessing

# IMPORTING THE LIBRARIES

# 'np' is the numpy shortcut!
# 'plt' is the matplotlib shortcut!
# 'pd' is the pandas shortcut!

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 1) Data preprocessing

# IMPORTING THE DATASET

# Data set is creating the data frame of the 'Position_Salaries.csv' file
# Features (independent variables) = The columns to predict the dependent variable
# Dependent variable = The last column
# 'X' = The matrix of features (country, age, salary)
# 'Y' = Dependent variable vector (purchased (last column))
# '.iloc' = locate indexes[rows, columns]
# ':' = all rows (all range)
# ':-1' = Take all the columns except the last one
# '.values' = taking all the values

# 'X = dataset.iloc[:, 1:-1].values' (This will take the values from the second column (index 1) to the second from last on)

# 'y = dataset.iloc[:, -1].values' (NOTICE! .iloc[all the rows, only the last column])

# 'X' (This contains the 'x' axis in a vertical 2D array)
# 'y' (This contains the 'y' axis in a 1D horizontal vector)

# Note to self: Only the CSV file and column/row values need to be changed here!

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# TRAINING THE RANDOM FOREST REGRESSION MODEL ON THE WHOLE DATASET
# 'n_estimators' = Number of trees (each tree is an estimator)
# 'random_state' =
# 'regressor' = fit the 'regressor' object to the whole dataset (train the regressor on the dataset)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# PREDICTING A NEW RESULT

regressor.predict([[6.5]])
print("This is the predicted result ")
print(regressor.predict([[6.5]]))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# VISUALISING THE RANDOM FOREST REGRESSION RESULTS (HIGHER RESOLUTION)

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()