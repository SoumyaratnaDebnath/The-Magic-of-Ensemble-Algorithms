import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import *
from ensemble.gradientBoosted import GradientBoostedRegressor
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# Or use sklearn decision tree
from sklearn.tree import DecisionTreeRegressor

################### GradientBoostedClassifier ###################

X, y = make_regression(n_features=1, n_informative=1, noise=10, tail_strength=10, random_state=42)

plt.scatter(X[:, 0], y, color='purple')
plt.title('Randomly generated data')
plt.show()

criteria = 'squared_error'
reg = GradientBoostedRegressor(base_estimator = DecisionTreeRegressor, criterion = criteria, n_estimators=20, max_depth=3)

reg.fit(X, y)
y_hat = reg.predict(X)
reg.plot(X, y)

y = pd.Series(y)
y_hat = pd.Series(y_hat)

print('Criteria :', criteria)
print('RMSE     :', rmse(y_hat, y))
print('MAE      :', mae(y_hat, y))

