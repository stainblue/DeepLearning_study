import numpy as np
from sklearn.datasets import load_iris
from KNN import KNN

def Test(K, X_train, y_train, X_test, y_test, y_name):
    classifier = KNN(K, X_train, y_train)

    print("# obtain_majority_vote")
    for i in range(X_test.shape[0]):
        computed_class = classifier.obtain_majority_vote(X_test[i])
        print("Test Data Index: {test_data_index}, Computed class: {computed_class}, True class: {true_class}".format(test_data_index=i, computed_class=y_name[computed_class], true_class=y_name[y_test[i]]))

    print("# obtain_weighted_majority_vote")
    for i in range(X_test.shape[0]):
        computed_class = classifier.obtain_weighted_majority_vote(X_test[i])
        print("Test Data Index: {test_data_index}, Computed class: {computed_class}, True class: {true_class}".format(test_data_index=i, computed_class=y_name[computed_class], true_class=y_name[y_test[i]]))
    
iris = load_iris()

X = iris.data       # iris data input
y = iris.target     # iris target (label)
y_name = iris.target_names # iris target name

X_train = []
y_train = []
X_test = []
y_test = []

for i in range(X.shape[0]):
    if (i + 1) % 15 == 0:
        X_test.append(X[i])
        y_test.append(y[i])
    else:
        X_train.append(X[i])
        y_train.append(y[i])

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print(X_test.shape)
print(y_name.shape)

for i in range(1, 11):
    print("\nK = {}".format(i))
    Test(i, X_train, y_train, X_test, y_test, y_name)