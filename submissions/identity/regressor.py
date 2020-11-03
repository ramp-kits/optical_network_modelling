import numpy as np


class Regressor():
    """Dummy regressor returning the input value of the cascade"""
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        # Get the input power for every channel
        X_array = np.array([np.array(X_i) for X_i in X[:, 1]])
        return X_array
