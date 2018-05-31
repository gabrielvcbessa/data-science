import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

np.random.seed(45)


def step_function(t):
    if t >= 0:
        return 1
    return 0


def prediction(x, w, b):
    return step_function((np.matmul(x, w) + b)[0])


def perceptron_step(x, y, w, b, learn_rate = 0.01):
    for i in range(len(x)):
        x_i = x.iloc[i]
        predicted = prediction(x_i, w, b)
        sign = y.iloc[i, 0] - predicted
        w[0] += sign * learn_rate * x_i[0]
        w[1] += sign * learn_rate * x_i[1]
        b += sign * learn_rate

    return w, b


if __name__ == "__main__":
    data = pd.read_csv('data.csv')

    x = data.iloc[:, 0:2]
    y = data.iloc[:, 2:3]

    x_max = max(x.T[0])

    w = np.array(np.random.rand(2, 1))
    b = np.random.rand(1)[0] + x_max

    boundary_lines = []

    # Running perceptron
    for i in range(100):
        w, b = perceptron_step(x, y, w, b)
        boundary_lines.append((-w[0]/w[1], -b/w[1]))

    # Printing stuff
    for i in range(len(x)):
        plot = 'bo' if y.iloc[i, 0] else 'ro'
        plt.plot([x.iloc[i, 0]], [x.iloc[i, 1]], plot)

    plt.axis([-0.5, 1.5, -0.5, 1.5])
    x_vals = np.array(plt.xlim())

    step = 3

    for i in range(0, len(boundary_lines), step):
        intercept, slope = boundary_lines[i]
        y_vals = intercept[0] * x_vals + slope[0]
        plot = 'k-' if i + step >= len(boundary_lines) else 'g--'
        plt.plot(x_vals, y_vals, plot)

    plt.show()
