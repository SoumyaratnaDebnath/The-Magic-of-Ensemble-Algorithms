from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import threading
import time

class BaggingClassifier():
    # initializing the model parameters
    def __init__(self, base_model=DecisionTreeClassifier, num_estimators=10, max_depth = 11, criterion = 'gini'):
        self.base_model = base_model
        self.num_estimators = num_estimators
        self.estimators = []
        self.fit_time = 0
        self.max_depth = max_depth
        self.criterion = criterion

    def fit(self, X, y, flag="sequencial"):
        if(flag == "parallel"):
            threads, models = [], []
            for i in range(self.num_estimators): models.append(self.base_model(max_depth = self.max_depth, criterion = self.criterion))
            for i in range(self.num_estimators): threads.append(threading.Thread(target=models[i].fit, args=(X[i], y[i],)))
            
            start_time = time.time()
            for i in range(self.num_estimators): threads[i].start()
            for i in range(self.num_estimators): threads[i].join()
            end_time = time.time()
            
            for i in range(self.num_estimators): self.estimators.append(models[i])
            self.fit_time = end_time - start_time

        elif(flag == "sequencial"):
            models = []
            for i in range(self.num_estimators): models.append(self.base_model(max_depth = self.max_depth, criterion = self.criterion))
            
            start_time = time.time()
            for i in range(self.num_estimators):
                models[i].fit(X[i], y[i])
            end_time = time.time()
            
            for i in range(self.num_estimators): self.estimators.append(models[i])
            self.fit_time = end_time - start_time
    
    def predict(self, X):
        predictions = np.zeros(len(X))
        for estimator in self.estimators:
            predictions += estimator.predict(X)
        return np.round(predictions / len(self.estimators))

    def plot(self, X, y):
        # Define the mesh grid for plotting
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Specifying the color map
        color_map = plt.get_cmap('bwr')

        # Plot the decision surface of each estimator
        fig, axs = plt.subplots(1, self.num_estimators, figsize=(self.num_estimators * 3, 3), sharex=True, sharey=True)
        for i, clf in enumerate(self.estimators):
            # Create a subplot for the current estimator
            ax = axs[i]

            # Plot the decision surface
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, alpha=0.4, cmap=color_map)

            # Plot the training data
            ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k', cmap=color_map)

            # Set the title of the subplot
            ax.set_title(f'Estimator {i+1}')

        # Show the plot
        plt.show()
        fig1 = plt

        # Plot the combined decision surface

        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.4, cmap=color_map)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k', cmap=color_map)
        plt.title('Combined decision surface')

        # Show the plot
        plt.show()
        fig2 = plt

        return [fig1, fig2]