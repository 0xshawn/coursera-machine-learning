import matplotlib.pyplot as plt
import numpy as np


def warm_up_exercise():
    print('warm_up_exercise')
    return np.eye(5)


def compute_cost(X, y, theta):
    m = y.shape[0]
    J = np.sum(np.square(X.dot(theta) - y)) / (2 * m)

    return J


def gradient_descent(X, y, theta, alpha, iterations):
    m = y.shape[0]
    j_history = np.zeros(iterations)

    for i in range(iterations):
        theta = theta - alpha * np.sum(X.transpose().dot(X.dot(theta) - y)) / m
        print('gradient_descent theta', theta)
        # theta = theta - (alpha/m)*np.sum((np.dot(X,theta)-y)[:,None]*X,axis=0)
        # theta = theta - alpha * (X' * (X * theta - y)) / length(y);

        j_history[i] = compute_cost(X, y, theta)

    return (theta, j_history)


if __name__ == '__main__':
    print(warm_up_exercise())

    data = np.array(np.loadtxt("ex1data1.txt", dtype='float64', delimiter=','))

    plt.plotfile('ex1data1.txt', delimiter=',', cols=(0, 1),
                 names=('Population of City in 10,000s', 'Profit in $10,000s'), marker='x', linestyle='',
                 color='r')
    # plt.show()

    x = data[:, :1]
    y = data[:, 1:2]

    ones = np.transpose([np.ones(x.shape[0])])
    X = np.hstack((ones, x))

    theta = np.transpose([np.zeros(2)])
    print('\ntheta initial value:')
    print(theta)

    iterations = 1500
    alpha = 0.01

    print('\nTest compute_cost')
    print('theta is [0; 0] and compute_cost is ', compute_cost(X, y, theta))
    print('Expected: 32.07')
    print('theta is [-1; 2] and compute_cost is ', compute_cost(X, y, np.array([[-1], [2]])))
    print('Expected: 54.24')

    theta, hist = gradient_descent(X, y, theta, alpha, iterations)
