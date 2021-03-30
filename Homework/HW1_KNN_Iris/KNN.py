import numpy as np

class KNN:
    def __init__(self, K, X_train, y_train):
        # K : K
        # X : Features
        # y : Target
        self.K = K
        self.X = X_train
        self.y = y_train

    def calculate_distance(self, X_test):
        # 여기서 self.X.shape = (# of train data, # of feature) = (15, 4) 이고, X_shape = (# of feature, ) = (4, ) 이다. 
        # 따라서 self.X - X_test 를 계산할 때, X_shape가 (15, 4)의 형태로 확장되어 계산된다.
        # X_test로부터 다른 각 X_train으로의 거리를 구해야하기 때문에, np.sum()을 계산할 때 axis=1 옵션을 주어 
        # 15 x 4 총 60개의 숫자의 합을 구하는 것이 아닌, 4개의 숫자의 합을 15번 구할 수 있게 한다.
        # 결과적으로 (15, ) 형태의 결과값을 리턴할 수 있게 한다.
        return np.sqrt(np.sum((self.X - X_test)**2, axis=1))
    
    def obtain_k_nearest_neighbor(self, X_test):
        calculated_distance = self.calculate_distance(X_test)
        
        # calculated_distance 를 굳이 모두 정렬할 필요 없이, 가장 값이 작은 (거리가 가까운) 점을 self.K 개를 구하면 된다.
        # np.argpartition() 함수를 이용하면 calculated_distance에서 가장 값이 작은 self.K 개의 값의 index를 구할 수 있다.
        # https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html
        k_neighbors_indices = np.argpartition(calculated_distance, self.K)[:self.K]

        # 그리고 이를 self.y에 마스킹하면 각 점의 클래스를, calculated_distance에 마스킹하면 각 점까지의 거리를 구할 수 있다.
        # 각 점까지의 거리는 obtain_weighted_majority_vote() 에서 weight 역할을 한다.
        k_neighbors_class = self.y[np.argpartition(calculated_distance, self.K)[:self.K]]
        k_neighbors_distance = calculated_distance[np.argpartition(calculated_distance, self.K)[:self.K]]

        return k_neighbors_class, k_neighbors_distance
        
        
    def obtain_weighted_majority_vote(self, X_test):
        return

    def obtain_majority_vote(self, X_test):
        return
    

from sklearn.datasets import load_iris

iris = load_iris()
# print(iris)

X = iris.data       # iris data input
y = iris.target     # iris target (label)
y_name = iris.target_names # iris target name

X_train, y_train = X[:15], y[:15]
X_test, y_test = X[15], y[15]
print("X_train : " + str(X_train))
print("y_train : " + str(y_train))
print("X_test : " + str(X_test))
print("y_test : " + str(y_test))

classifier = KNN(3, X_train, y_train)
classifier.obtain_k_nearest_neighbor(X_test)