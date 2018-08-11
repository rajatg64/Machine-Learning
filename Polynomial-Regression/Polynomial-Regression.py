
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#inmport the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

#Split into test and training set


#fitting into linear regression

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

# fitting into polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, Y)



#Linear regression output

plt.scatter(X, Y, color = "red")
plt.scatter(X, lin_reg.predict(X), color = "blue")
plt.title("Truth or Bluff(Linear Regression)")
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()

#Polynomial Regression output
x_grid = np.arange(min(X),  max(Y), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)


plt.scatter(X, Y, color = "red")
plt.scatter(x_grid, lin_reg2.predict(poly_reg.fit_transform(x_grid)), color = "green")
plt.title("Truth or Bluff(Ploynomial Regression)")
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()


#prediction

lin_reg.predict(6.5)
lin_reg2.predict(poly_reg.fit_transform(6.5))