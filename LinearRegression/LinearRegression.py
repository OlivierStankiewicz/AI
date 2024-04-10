import numpy as np
import matplotlib.pyplot as plt
import random

from data import get_data, inspect_data, split_data

data = get_data()
#inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2     ZLY WZOR, brakuje 1/m

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution
def calcThetaBest(x, y):
    x = np.asmatrix(x)
    x = x.transpose()
    m = x.shape[0]
    x = np.append(x, np.ones((m, 1)), axis=1)

    y = np.asmatrix(y)
    y = y.transpose()

    theta = np.dot(np.linalg.inv(np.dot(x.transpose(), x)), np.dot(x.transpose(), y))
    theta = np.asarray(theta)
    return [theta[1][0], theta[0][0]]

theta_best = calcThetaBest(x_train, y_train)
print(f'Theta = {theta_best}')

# TODO: calculate error
def calcError(theta, x, y):
    x = list(x)
    m = len(x)
    for i in range(m):
        x[i] = [1, x[i]]
    y = list(y)

    sum = 0
    for i in range(m):
        sum += (theta[0]*x[i][0] + theta[1]*x[i][1] - y[i])**2

    return sum/m

def calcError2(theta, x, y):
    x = list(x)
    m = len(x)
    for i in range(m):
        x[i] = [1, x[i]]
    y = list(y)

    sum = 0
    for i in range(m):
        sum += (theta[0]*x[i][0] + theta[1]*x[i][1] + theta[2]*x[i][1]**2 - y[i])**2

    return sum/m

MSE_theta = calcError(theta_best, x_test, y_test)
print(f'MSE(theta) = {MSE_theta}')

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization
mean_x = np.mean(x_train)
deviation_x = np.std(x_train)
x_train = (x_train - mean_x) / deviation_x
x_test = (x_test - mean_x) / deviation_x

mean_y = np.mean(y_train)
deviation_y = np.std(y_train)

# TODO: calculate theta using Batch Gradient Descent
def calcGradient(theta, x, y):
    m = x.shape[0]
    tmp = np.ones((m,3))
    tmp[:,1] = x
    tmp[:,2] = x**2
    x = tmp

    return 2*np.dot(x.transpose(), (np.matmul(x, theta) - y))/m

theta_best = np.asarray([random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)])
learning_rate = 0.01
theta_new = theta_best - learning_rate*calcGradient(theta_best, x_train, y_train)

error2 = calcError2(theta_best, x_train, y_train)
train_MSE = []

while True:
    theta_best = theta_new
    theta_new = theta_best - learning_rate*calcGradient(theta_best, x_train, y_train)
    error1 = error2
    error2 = calcError2(theta_best, x_train, y_train)
    train_MSE.append(error2)
    if error1<=error2:
        break

print(f'Theta = {theta_best}')

# TODO: calculate error
MSE_theta = calcError2(theta_best, x_test, y_test)
print(f'MSE(theta) = {MSE_theta}')

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x + float(theta_best[2]) * x**2
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

plt.plot(train_MSE)
plt.xlabel('epoki')
plt.ylabel('train MSE')
plt.show()