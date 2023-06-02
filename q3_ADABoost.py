import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import *
from ensemble.ADABoost import AdaBoostClassifier
# from tree.base import DecisionTree
# Or you could import sklearn DecisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

################### AdaBoostClassifier on Real Input and Discrete Output ###################

N = 100
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3

# Generating random data
X, y = make_classification(n_samples=N, n_features=P, n_informative=P, n_redundant=0, random_state=4, n_classes=NUM_OP_CLASSES)
plt.scatter(X[:,0], X[:,1], c=y, cmap = plt.get_cmap('bwr'))
plt.title("Dataset generated randomly")
plt.show()

criteria = 'gini'
Classifier_AB = AdaBoostClassifier(base_model=DecisionTreeClassifier, criterion=criteria, n_estimators=n_estimators)
Classifier_AB.fit(X, y)
y_hat = Classifier_AB.predict(X)
[fig1, fig2] = Classifier_AB.plot(X, y)

y = pd.Series(y, dtype="category")
y_hat = pd.Series(y_hat, dtype="category")

print("Criteria  :", criteria)
print("Accuracy  :", accuracy(y_hat, y))
for cls in y.unique():
    print('\nClass     :', cls)
    print("Precision :", precision(y_hat, y, cls))
    print("Recall    :", recall(y_hat, y, cls))