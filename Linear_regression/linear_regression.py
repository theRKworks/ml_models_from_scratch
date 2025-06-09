import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

def error_fuction(x,y,a,b):
    error = 0
    for i in range(len(x)):
        error += (y[i] - (a*x[i] + b))**2
    return error / len(x)

def gradient_descent(x, y, a, b, L):
    a_gradient = 0
    b_gradient = 0
    N = float(len(x))
    for i in range(len(x)):
        a_gradient += (-2/N) * x[i] * (y[i] - (a * x[i] + b))
        b_gradient += (-2/N) * (y[i] - (a * x[i] + b))
    a -= L * a_gradient
    b -= L * b_gradient
    return a, b

def linear_regression(x, y, L=0.01, epochs=1000):
    a = 0
    b = 0
    for i in range(epochs):
        a, b = gradient_descent(x, y, a, b, L)
        if i % 100 == 0:
            print(f"Epoch {i}: Error = {error_fuction(x, y, a, b)}")
    return a, b

def plot_regression_line(x, y, a, b):
    plt.scatter(x, y, color='blue', label='Data points')
    plt.plot(x, a * x + b, color='red', label='Regression line')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Linear Regression')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Generate synthetic data
    x, y = make_regression(n_samples=100, n_features=1, noise=10)
    x = x.flatten()

    # Train linear regression model
    a, b = linear_regression(x, y, epochs=300)

    # Plot the regression line
    plot_regression_line(x, y, a, b)

