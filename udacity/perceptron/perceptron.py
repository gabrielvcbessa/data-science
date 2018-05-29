import pandas as pd
import numpy as np

np.random.seed(25)


def stepFunction(t):
    if t >= 0:
        return 1
    return 0


def prediction(x, w, b):
    return stepFunction((np.matmul(x, w) + b)[0])


def perceptron_step(x, y, w, b, learn_rate=0.01):
    for i in range(len(x)):
        x_i = x.iloc[i]
        predicted = prediction(x_i, w, b)
        sign = y.iloc[i] - predicted
        w[0] += sign * learn_rate * x_i[0]
        w[1] += sign * learn_rate * x_i[1]
        b += sign * learn_rate

    return w, b


if __name__ == "__main__":
    data = pd.read_csv('data.csv')

    x = data.iloc[:, 0:2]
    y = data.iloc[:, 2:]

    w = np.array(np.random.rand(2, 1))
    b = np.random.rand(1)[0]

    best_result = None

    for i in range(25):
        w, b = perceptron_step(x, y, w, b, 0.1)
        best_result = (w, b)

    print(best_result)