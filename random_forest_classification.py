import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import *
from tree.randomForest import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification, make_regression

np.random.seed(42)

N = 100
P = 3
num_classes = 2
n_estimators = 3

X, y = make_classification(n_samples=N, n_features=P, n_informative=P, n_redundant=0, random_state=42, n_classes=num_classes)
plt.scatter(X[:,0], X[:,1], c=y, cmap = plt.get_cmap('bwr'))
plt.title("Dataset generated randomly")
plt.show()

criteria = 'gini'
clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=5, max_features=2, method='column_sampling', criterion=criteria)
clf.fit(X, y)
y_pred = clf.predict(X)
clf.plot()

y_pred = pd.Series(y_pred)
y = pd.Series(y)

print('Criteria  :', criteria)
print('Accuracy  :', accuracy(y_pred, y))
for cls in y.unique():
  print('\nClass', cls)
  print('Precision :', precision(y_pred, y, cls))
  print('Recall    :', recall(y_pred, y, cls))




