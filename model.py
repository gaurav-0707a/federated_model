
import numpy as np
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def train_step(X, y, weights, learning_rate=0.01):
    predictions = sigmoid(np.dot(X, weights))
    error = predictions - y
    gradient = np.dot(X.T, error) / len(y)
    return gradient