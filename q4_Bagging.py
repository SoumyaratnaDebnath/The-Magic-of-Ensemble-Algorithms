from sklearn.datasets import make_classification
from metrics import *
from ensemble.bagging import BaggingClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Or use sklearn decision tree

from sklearn.tree import DecisionTreeClassifier

np.random.seed(42)

################### BaggingClassifier ###################

N = 50
P = 2
NUM_OP_CLASSES = 2
n_estimators = 5
max_depth = 20

# Generate a synthetic dataset for classification
X, y = np.random.randn(N, P), np.random.randint(NUM_OP_CLASSES, size=N)
plt.scatter(X[:,0], X[:,1], c=y, cmap = plt.get_cmap('bwr'))
plt.title("Dataset generated randomly")
plt.show()

# creating dataset samples for tarining
X_params, y_params = [], []
for i in range(n_estimators):
    sample_indices = np.random.choice(len(X), len(X), replace=True)
    sample_indices = sample_indices[:int(sample_indices.size/2)]
    X_params.append(X[sample_indices])
    y_params.append(y[sample_indices])

criteria = "gini"

# sequencial training
classifier_B = BaggingClassifier(base_model=DecisionTreeClassifier, num_estimators=n_estimators, criterion = criteria, max_depth = max_depth)
classifier_B.fit(X_params, y_params, flag="sequencial")
y_hat = classifier_B.predict(X)

[fig1, fig2] = classifier_B.plot(X, y)

y = pd.Series(y, dtype="category")
y_hat = pd.Series(y_hat, dtype="category")

print("Criteria   :", criteria)
print("Accuracy   :", accuracy(y_hat, y))
for cls in y.unique():
    print("\nClass      :", cls)
    print("Precision  :", precision(y_hat, y, cls))
    print("Recall     :", recall(y_hat, y, cls))


############## For Comparison ##############
print("\n\nFor comparison between sequencial and parallel training\n")

N = 2000
P = 2
NUM_OP_CLASSES = 4
n_estimators = 20
max_depth = 20

# Generate a synthetic dataset for classification
X, y = np.random.randn(N, P), np.random.randint(NUM_OP_CLASSES, size=N)
plt.scatter(X[:,0], X[:,1], c=y, cmap = plt.get_cmap('bwr'))
plt.title("Dataset generated randomly")
plt.show()

# creating dataset samples for tarining
X_params, y_params = [], []
for i in range(n_estimators):
    sample_indices = np.random.choice(len(X), len(X), replace=True)
    X_params.append(X[sample_indices])
    y_params.append(y[sample_indices])

criteria = "gini"

# sequencial training
classifier_seq = BaggingClassifier(base_model=DecisionTreeClassifier, num_estimators=n_estimators, criterion = criteria, max_depth = max_depth)
classifier_seq.fit(X_params, y_params, flag="sequencial")
y_hat_seq = classifier_seq.predict(X)

# parallel training
classifier_par = BaggingClassifier(base_model=DecisionTreeClassifier, num_estimators=n_estimators, criterion = criteria, max_depth = max_depth)
classifier_par.fit(X_params, y_params, flag="parallel")
y_hat_par = classifier_par.predict(X)

y = pd.Series(y, dtype="category")
y_hat_seq = pd.Series(y_hat_seq, dtype="category")
y_hat_par = pd.Series(y_hat_par, dtype="category")

print("Criteria            :", criteria)
print("Accuracy Sequencial :", accuracy(y_hat_seq, y))
print("Accuracy Parallel   :", accuracy(y_hat_par, y))

print("\nTime elapsed in sequencial execution :", round(classifier_seq.fit_time, 4), "seconds")
print("Time elapsed in parallel execution   :", round(classifier_par.fit_time, 4), "seconds")
