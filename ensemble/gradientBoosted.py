import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

class GradientBoostedRegressor:
    def __init__(self, base_estimator, n_estimators=3, learning_rate=0.1, max_depth=2, criterion='squared_error'):  
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.alphas = []
        self.avg_pred = 0
        self.max_depth = max_depth
        self.criterion = criterion

    def fit(self, X, y):
        y_residual = y.copy()
        self.avg_pred = np.mean(y)
        y_residual = y_residual - self.avg_pred
        for _ in range(self.n_estimators):
            model = self.base_estimator(max_depth=self.max_depth, criterion=self.criterion)
            model.fit(X, y_residual)
            y_pred = model.predict(X)
            y_residual = y_residual - self.learning_rate*(y_pred)
            self.models.append(model)
            self.alphas.append(self.learning_rate)

    def predict(self, X):
        model_preds = np.array([model.predict(X) for model in self.models])
        return self.avg_pred + np.dot(self.alphas, model_preds)

    def plot(self, X, y):
        y_pred = self.predict(X)
        plt.plot(X, y, 'o', color='purple', label='Actual (y)')
        plt.plot(X, y_pred, 'o', color='orange', label='Predicted (y_hat)')
        plt.title('Prediction')
        plt.legend()
        plt.show()