import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# generating the dataset
np.random.seed(1234)
x = np.linspace(0, 10, 50)
eps = np.random.normal(0, 5, 50)
y = x**2 +1 + eps

#for plotting
plt.plot(x, y, 'o')
plt.plot(x, x**2 + 1, 'r-')
plt.show()

# printning number of datapoints and labels
print('No of datapoints = {0}, No. of Labels = {1}'.format(x.size, y.size))

# splitting the data into train and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)

print('x_train size = {0}, x_test size = {1}'.format(x_train.size, x_test.size))
print('y_train size = {0}, y_test size = {1}'.format(y_train.size, y_test.size))

# specifying the depth range
max_depths = range(1, 11)

# lists to store training and validation MSEs
train_mse = []
test_mse = []

bias = []
variance = []

# reshaping the array x_train and x_test for training
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)
x = x.reshape(-1, 1)

# decision tree regressor for each depth and compute the MSEs
criterion = 'squared_error'
for depth in max_depths:
    model = DecisionTreeRegressor(max_depth=depth, criterion=criterion)
    model.fit(x_train, y_train)

    y_cap_train = model.predict(x_train)
    y_cap_test = model.predict(x_test)
    y_cap = model.predict(x)

    mse_train = np.mean((y_cap_train - y_train) ** 2)
    mse_test = np.mean((y_cap_test - y_test) ** 2)
    
    bias.append(np.mean(y_test - np.mean(y_cap_test)))
    variance.append(np.mean((y_cap_test - np.mean(y_cap_test))**2))

    train_mse.append(mse_train)
    test_mse.append(mse_test)

bias = np.array(bias)
variance = np.array(variance)
bias = (bias - bias.min())/(bias.max() - bias.min())
variance = (variance - variance.min())/(variance.max() - variance.min())

# Plot the bias-variance tradeoff as a function of tree depth
plt.plot(max_depths, train_mse, label='Training MSE', color='blue')
plt.plot(max_depths, test_mse, label='Testing MSE', color='red')
plt.xlabel('Max Depth')
plt.ylabel('Mean Squared Error')
plt.title('Training and Testing Curve (MSE vs Depth)')
plt.legend()
plt.grid()
plt.show()

# Plot the bias-variance tradeoff as a function of tree depth
plt.plot(max_depths, bias, label='Bias', color='blue')
plt.plot(max_depths, variance, label='Variance', color='red')
plt.title('Bias-Variance Tradeoff in Decision Tree Regression')
plt.xlabel('Max Depth')
plt.ylabel('Normalised Error')
plt.legend()
plt.grid()
plt.show()