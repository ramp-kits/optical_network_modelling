import numpy as np
from sklearn.linear_model import Ridge


class Regressor:
    def __init__(self):
        pass

    def fit(self, X, y):
        X_array = np.array([np.array(X_i) for X_i in X[:, 1]])
        self.reg = Ridge(alpha=1)
        self.reg.fit(X_array, y)

    def predict(self, X):
        X_array = np.array([np.array(X_i) for X_i in X[:, 1]])
        return self.reg.predict(X_array)
