# Importing necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Managing the view of the data in pandas

desired_width = 175
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 15)

# Getting the data from the external file using pandas

customers = pd.read_csv(r'C:\Users\miscasrikanth\Desktop\Data science\Refactored_Py_DS_ML_Bootcamp-master\11-Linear-Regression\Ecommerce Customers')

# Insights on the data

print(customers.head())
print(customers.describe())
print(customers.info())

# EDA on the data

sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=customers)
sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=customers)
sns.jointplot(x='Time on App', y='Length of Membership', data=customers, kind="hex")
sns.pairplot(customers)
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers)


# Splitting the data in training and testing

print(customers.columns)

X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
Y = customers['Yearly Amount Spent']

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)

# Training the model

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, Y_train)

# Printing the Coefficients

print(lm.coef_)

# Predicting the data

predictions = lm.predict(X_test)

sns.scatterplot(x=Y_test, y=predictions)

# Evaluating the metrics

from sklearn import metrics

print('MAE-->', metrics.mean_absolute_error(Y_test, predictions))
print('MSE-->', metrics.mean_squared_error(Y_test, predictions))
print('RMSE-->', np.sqrt(metrics.mean_squared_error(Y_test, predictions)))
print('Explained Variance-->', metrics.explained_variance_score(Y_test, predictions))

# Plotting the residuals

sns.displot((Y_test-predictions), bins=50)
plt.show()

# Analysis of Data

Coefficients = pd.DataFrame(lm.coef_, X.columns)
Coefficients.columns = ['Coefficient']
print(Coefficients)
