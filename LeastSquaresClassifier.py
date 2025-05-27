import numpy as np

class LeastSquaresClassifier:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        # Add bias
        X_bias = np.hstack([X, np.ones((X.shape[0], 1))])
        # Minimize error
        self.weights = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y

    def predict(self, X):
        X_bias = np.hstack([X, np.ones((X.shape[0], 1))])
        preds = X_bias @ self.weights
        return (preds >= 0.5).astype(int)  # threshold = 0.5
