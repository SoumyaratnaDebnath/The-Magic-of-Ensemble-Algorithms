from numpy.core.fromnumeric import shape
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd

class RandomForestClassifier():
    def __init__(self, n_estimators=3, criterion='gini', max_depth=5, method='row_sampling', max_features=2):
        self.max_depth=max_depth
        self.n_estimators=n_estimators
        self.estimators_=[None]*n_estimators

        self.max_features = max_features
        self.trees = []
        self.features = []
        self.feature_sets = []
        self.criterion = criterion

        self.method = method

    def fit(self, X, y):
      if(self.method == 'row_sampling'):
        X_temp1=X.copy()
        X_temp1["res"]=y
        for i in range(self.n_estimators):
            X_temp=X_temp1.sample(frac=0.6)
            d_tree = DecisionTreeClassifier(max_depth=self.max_depth, criterion=self.criterion)
            d_tree.fit(X_temp.iloc[:,:-1],X_temp.iloc[:,-1])
            self.estimators_[i]=d_tree
      if(self.method == 'column_sampling'):
        n_features = X.shape[1]
        for i in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth, criterion=self.criterion)
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
            X_subset = X[:, feature_indices]
            tree.fit(X_subset, y)
            self.trees.append(tree)
            self.features.append(feature_indices)
            self.feature_sets.append([X_subset, y])
            self.estimators_[i] = tree

    def predict(self, X):
      if(self.method == 'row_sampling'):
        res=np.zeros((X.shape[0],self.n_estimators))
        for i in range(self.n_estimators):
            d_tree=self.estimators_[i]
            res[:,i]=np.array(d_tree.predict(X))
        y_hat=np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            a=list(res[i])
            y_hat[i]=max(set(a),key=a.count)
        return pd.Series(y_hat)

      if(self.method == 'column_sampling'):
        predictions = np.zeros((X.shape[0], len(self.trees)))
        for i, (tree, feature_indices) in enumerate(zip(self.trees, self.features)):
            X_subset = X[:, feature_indices]
            predictions[:, i] = tree.predict(X_subset)
        return np.mean(predictions, axis=1)

    def plot(self):
        
        n_estimators = self.n_estimators
        # Create a figure with subplots for each tree
        fig, axes = plt.subplots(1, n_estimators, figsize=(5 * n_estimators, 3))
        # Loop over each tree in the random forest
        for i in range(n_estimators):
            t = self.estimators_[i]
            # Plot the decision surface for the ith tree
            tree.plot_tree(t, ax=axes[i], filled=True, rounded=True, class_names=True)
            axes[i].set_title(f"Tree {i+1}")
        plt.show()

        if(self.method == 'column_sampling'):
          fig, axs = plt.subplots(1, self.n_estimators, figsize=(self.n_estimators * 5, 3), sharex=True, sharey=True)
          for i, clf in enumerate(self.trees):
            ax = axs[i]
            title = "Estimator "+str(i+1)
            self.plot_decision_surfaces(clf, ax, self.feature_sets[i][0], self.feature_sets[i][1], title)
          plt.show()

          fig, axs = plt.subplots(1, 1, figsize=(5, 5), sharex=True, sharey=True)
          for i, clf in enumerate(self.trees):
            ax = axs
            title = "Combined decision surface"
            self.plot_decision_surfaces(clf, ax, self.feature_sets[i][0], self.feature_sets[i][1], title)
          plt.show()


    def plot_decision_surfaces(self, clf, ax, X, y, title):
        color_map = plt.get_cmap('bwr')

        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.4, cmap=color_map)
        ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k', cmap=color_map)
        ax.set_title(title)
      

class RandomForestRegressor():
    def __init__(self, n_estimators=3, criterion='squared_error', max_depth=5, method='row_sampling', max_features=2):
        self.n_estimators=n_estimators
        self.estimators_=[None]*n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.method = method

        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
      if(self.method == 'row_sampling'):
        X_temp1=X.copy()
        X_temp1["res"]=y
        for i in range(self.n_estimators):
            X_temp=X_temp1.sample(frac=0.6)
            d_tree = DecisionTreeRegressor(max_depth = self.max_depth, criterion=self.criterion)
            d_tree.fit(X_temp.iloc[:,:-1],X_temp.iloc[:,-1])
            self.estimators_[i]=d_tree

      if(self.method == 'column_sampling'):
        n_features = X.shape[1]
        if self.max_features is None:
            self.max_features = n_features
        for i in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth, criterion=self.criterion)
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
            X_subset = X[:, feature_indices]
            tree.fit(X_subset, y)
            self.trees.append((tree, feature_indices))
            self.estimators_[i]=tree

    def predict(self, X):
      if(self.method == 'row_sampling'):
        res=np.zeros((X.shape[0],self.n_estimators))
        for i in range(self.n_estimators):
            d_tree=self.estimators_[i]
            res[:,i]=np.array(d_tree.predict(X))
        y_hat=np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_hat[i]=np.mean(res[i])
        return pd.Series(y_hat)

      if(self.method == 'column_sampling'):
        predictions = np.zeros((X.shape[0], len(self.trees)))
        for i, (tree, feature_indices) in enumerate(self.trees):
            X_subset = X[:, feature_indices]
            predictions[:, i] = tree.predict(X_subset)
        return np.mean(predictions, axis=1)

    def plot(self):
      n_estimators = self.n_estimators
      # Create a figure with subplots for each tree
      fig, axes = plt.subplots(1, n_estimators, figsize=(5 * n_estimators, 3))
      # Loop over each tree in the random forest
      for i in range(n_estimators):
          t = self.estimators_[i]
          # Plot the decision surface for the ith tree
          tree.plot_tree(t, ax=axes[i], filled=True, rounded=True, class_names=True)
          axes[i].set_title(f"Tree {i+1}")
      plt.show()




