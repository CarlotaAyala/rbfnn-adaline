
import numpy as np

class Adaline:
    def __init__(self, eta=0.0005, epochs=500):
        self.eta = eta
        self.epochs = epochs
        self.W = None
        self.errors = []
        
    def predict(self, X):
        return np.dot(X, self.W[1:]) + self.W[0]

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.W = np.random.normal(loc=0.0, scale=0.01, size=1 + n_features)
        for _ in range(self.epochs):
            output = self.predict(X)
            errors = y.flatten() - output
            self.W[1:] += self.eta * X.T.dot(errors)
            self.W[0] += self.eta * errors.sum()
            mse = (errors ** 2).mean()
            self.errors.append(mse)
        return self
