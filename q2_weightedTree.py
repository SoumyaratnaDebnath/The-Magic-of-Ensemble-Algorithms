from tree.base import WeightedDecisionTree
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from metrics import *
import matplotlib.pyplot as plt


X, y = make_classification(
n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)
X = pd.DataFrame(X)
y = pd.Series(y, dtype = 'category')

# X['Weight'] = [1/len(X)]*len(X)
# sample_weight = pd.Series(np.ones(len(X)), dtype = 'float64')
# assign weights in uniform distribution between 0 and 1

sample_weight = pd.Series(np.random.uniform(0,1, len(X)), dtype = 'float64')
# sample_weight = sample_weight.apply(lambda x: np.random.uniform(0,1))
print(X.head())

# shuffe the X, y and sample_weight in the same order
df = pd.concat([X, y, sample_weight], axis = 1)
df = df.sample(frac=1).reset_index(drop=True)
# X = df.iloc[:, 0:2]
# y = df.iloc[:, 2]
# sample_weight = df.iloc[:, 3]
X = df.iloc[:, 0:-2]
y = df.iloc[:, -2]
sample_weight = df.iloc[:, -1]


print(X.shape, y.shape, sample_weight.shape)
X_train, X_test  = pd.DataFrame(X[:int(0.7*len(X))]), pd.DataFrame(X[int(0.7*len(X)):])
y_train, y_test = pd.Series(y[:int(0.7*len(y))], dtype = 'category'), pd.Series(y[int(0.7*len(y)):], dtype = 'category')
wt_train, wt_test = pd.Series(sample_weight[:int(0.7*len(y))]), pd.Series(sample_weight[int(0.7*len(y)):])

print("X_train.shape, X_test.shape, y_train.shape, y_test.shape")
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print("wt_train.shape, wt_test.shape")
print(wt_train.shape, wt_test.shape)

color_map = plt.get_cmap('bwr')

# Visulaize the data splits 
plt.scatter(X_train[0], X_train[1], c=y_train, cmap=color_map)
# save and show the plot
plt.title('Train Data')
# plt.savefig('./q2/q2_train_data.png')
plt.show()


plt.scatter(X_test[0], X_test[1], c=y_test, cmap=color_map)
# save and show the plot
plt.title('Test Data')
# plt.savefig('./q2/q2_test_data.png')
plt.show()


def my_plot(X,y, tree, criteria, tree_name):
    X_np = X.to_numpy()
    y_np = y.to_numpy()

    min1, max1 = X_np[:, 0].min() - 1, X_np[:, 0].max() + 1
    min2, max2 = X_np[:, 1].min() - 1, X_np[:, 1].max() + 1
    # define the x and y scale
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)
    # create all of the lines and rows of the grid
    xx, yy = np.meshgrid(x1grid, x2grid)
    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    # horizontal stack vectors to create x1,x2 input for the model
    grid = np.hstack((r1,r2))
    # make predictions for the grid
    model = tree
    yhat = model.predict(pd.DataFrame(grid))
    if tree_name != 'Sklearn Decision Tree':
        yhat = yhat.to_numpy()
    # reshape the predictions back into a grid
    zz = yhat.reshape(xx.shape)
    # plot the grid of x, y and z values as a surface
    # plt.contourf(xx, yy, zz, cmap='Paired', alpha=0.3)
    plt.contourf(xx, yy, zz, cmap=color_map)
    # create scatter plot for samples from each class
    for class_value in range(len(y.unique())):
        # get row indexes for samples with this class
        row_ix = np.where(y_np == class_value)
        # create scatter of these samples
        # plt.scatter(X_np[row_ix, 0], X_np[row_ix, 1], label = class_value)
        plt.scatter(X_np[row_ix, 0], X_np[row_ix, 1], label = class_value, cmap=color_map)
    # show the plot
    # add title and labels
    plt.title(f'Decision Tree split on {criteria} with {tree_name}')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    # plt.savefig(f'./q2/q2_decision_tree_{criteria} with {tree_name}.png')
    plt.show()


for criteria in ['information_gain', 'gini_index']:
    tree = WeightedDecisionTree(criterion=criteria, max_depth=10) #Split based on Inf. Gain
    tree.fit(X_train, y_train, wt_train)
    # tree.plot()
    y_hat_train = tree.predict(X_train)
    print('Criteria :', criteria)
    print("\nTraining metrics")
    print('Accuracy: ', accuracy(y_hat_train, y_train))
    for cls in y.unique():
        print('\nClass: ', cls)
        print('\tPrecision: ', precision(y_hat_train, y_train, cls))
        print('\tRecall: ', recall(y_hat_train, y_train, cls))
    print("\nTesting metrics")
    y_hat_test = tree.predict(X_test)
    print('Accuracy: ', accuracy(y_hat_test, y_test.reset_index(drop=True)))
    for cls in y.unique():
        print('\nClass: ', cls)
        print('\tPrecision: ', precision(y_hat_test, y_test.reset_index(drop=True), cls))
        print('\tRecall: ', recall(y_hat_test, y_test.reset_index(drop = True), cls))

    print("\nSklearn metrics")
    if criteria == 'information_gain':
        sklearn_criteria = 'entropy'
    else:
        sklearn_criteria = 'gini'
    clf = DecisionTreeClassifier(criterion=sklearn_criteria, max_depth=10)
    clf.fit(X_train, y_train, sample_weight = wt_train)
    y_hat_train = clf.predict(X_train)
    print('Criteria :', criteria)
    print("\nTraining metrics")
    print('Accuracy: ', accuracy(y_hat_train, y_train))
    for cls in y.unique():
        print('\nClass: ', cls)
        print('\tPrecision: ', precision(y_hat_train, y_train, cls))
        print('\tRecall: ', recall(y_hat_train, y_train, cls))
    print("\nTesting metrics")
    y_hat_test = clf.predict(X_test)
    print('Accuracy: ', accuracy(y_hat_test, y_test.reset_index(drop=True)))
    for cls in y.unique():
        print('\nClass: ', cls)
        print('\tPrecision: ', precision(y_hat_test, y_test.reset_index(drop=True), cls))
        print('\tRecall: ', recall(y_hat_test, y_test.reset_index(drop = True), cls))



    my_plot(X,y, tree, criteria, 'Weighted Decision Tree')
    my_plot(X,y, clf, criteria, 'Sklearn Decision Tree')



