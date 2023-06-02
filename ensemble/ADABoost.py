
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class AdaBoostClassifier():
    def __init__(self, n_estimators=3, criterion = 'gini', base_model=DecisionTreeClassifier): # Optional Arguments: Type of estimator
        '''
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        '''
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []
        self.criterion = criterion
        self.base_model=base_model

    def fit(self, X, y):
        """
        Function to train and construct the AdaBoostClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        n_samples = X.shape[0]
        w = np.full(n_samples, (1 / n_samples))
        for i in range(self.n_estimators):
            model = self.base_model(max_depth=1, criterion=self.criterion)
            model.fit(X, y, sample_weight=w)
            y_pred = model.predict(X)
            error = np.sum(w[y_pred != y])
            alpha = 0.5 * np.log((1 - error) / error)
            w *= np.exp(-alpha * y * y_pred)
            w /= np.sum(w)
            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, X):
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        model_preds = np.array([model.predict(X) for model in self.models])
        return np.sign(np.dot(self.alphas, model_preds))

    def plot(self, X, y):
        """
        Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns
        The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
        Further, the scatter plot should have the marker size corresponnding to the weight of each point.

        Figure 2 should also create a decision surface by combining the individual estimators

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]
        """
        # Define the mesh grid for plotting
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Specifying the color map
        color_map = plt.get_cmap('bwr')

        # Plot the decision surface of each estimator
        fig, axs = plt.subplots(1, self.n_estimators, figsize=(self.n_estimators * 3, 3), sharex=True, sharey=True)
        for i, clf in enumerate(self.models):
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
