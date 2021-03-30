import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
# print(iris)

X = iris.data       # iris data input
y = iris.target     # iris target (label)
y_name = iris.target_names # iris target name

# print(X)
# print(y)
# print(y_name)

X = iris.data[:, :2]    # for now, use the first two features.
y = iris.target

x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5,      # first feature
x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5,      # second feature

plt.figure(figsize=(8, 6))
# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.Set1, edgecolor = 'k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)

plt.show()