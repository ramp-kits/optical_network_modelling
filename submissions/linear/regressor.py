import numpy as np
from sklearn.linear_model import Ridge


class Regressor:
    """A sklearn linear model"""
    def __init__(self):
        # Instanciate sklearn model
        self.reg = Ridge(alpha=1)

    def fit(self, X, y):
        # Get the input power for every channel
        X_array = np.array([np.array(X_i) for X_i in X[:, 1]])
        # Fit the model with the training data
        self.reg.fit(X_array, y)

    def predict(self, X):
        # Get the input power for every channel
        X_array = np.array([np.array(X_i) for X_i in X[:, 1]])
        # Return positive values when evaluating the model
        return np.maximum(0, self.reg.predict(X_array))
