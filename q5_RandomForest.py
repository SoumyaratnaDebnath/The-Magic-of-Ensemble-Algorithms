import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import *
from tree.randomForest import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification, make_regression

np.random.seed(42)

################### RandomForestClassifier ###################

N = 30
P = 5
num_classes = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(num_classes, size = N), dtype="category")

for criteria in ['entropy', 'gini']:
    Classifier_RF = RandomForestClassifier(criterion = criteria)
    Classifier_RF.fit(X, y)
    y_hat = Classifier_RF.predict(X)
    Classifier_RF.plot()
    print('Criteria  :', criteria)
    print('Accuracy  :', accuracy(y_hat, y))
    for cls in y.unique():
      print('\nClass', cls)
      print('Precision :', precision(y_hat, y, cls))
      print('Recall    :', recall(y_hat, y, cls))

################### RandomForestRegressor ###################

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

criteria = 'squared_error'
Regressor_RF = RandomForestRegressor(criterion = criteria)
Regressor_RF.fit(X, y)
y_hat = Regressor_RF.predict(X)
Regressor_RF.plot()
print('Criteria :', criteria)
print('RMSE     :', rmse(y_hat, y))
print('MAE      :', mae(y_hat, y))



N = 30
P = 3
n_estimators = 3

X, y = make_regression(n_samples=N, n_features=P, noise=0.2)

criteria = 'squared_error'
clf = RandomForestRegressor(n_estimators=n_estimators, max_depth=5, max_features=2, method='column_sampling', criterion=criteria)
clf.fit(X, y)
y_pred = clf.predict(X)
clf.plot()

y_pred = pd.Series(y_pred)
y = pd.Series(y)

print('Criteria :', criteria)
print('RMSE     :', rmse(y_pred, y))
print('MAE      :', mae(y_pred, y))
