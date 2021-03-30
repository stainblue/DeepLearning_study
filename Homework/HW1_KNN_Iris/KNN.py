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
        # 여기서 self.X.shape = (# of train data, # of feature) = (# of train data, 4) 이고, X_shape = (# of feature, ) = (4, ) 이다. 
        # 따라서 self.X - X_test 를 계산할 때, X_shape가 (# of train data, 4)의 형태로 확장되어 계산된다.
        # X_test로부터 다른 각 X_train으로의 거리를 구해야하기 때문에, np.sum()을 계산할 때 axis=1 옵션을 주어 
        # # of train data x 4 의 숫자의 합을 구하는 것이 아닌, 4개의 숫자의 합을 # of train data번 구할 수 있게 한다.
        # 결과적으로 (# of train data, ) 형태의 결과값을 리턴할 수 있게 한다.
        return np.sqrt(np.sum((self.X - X_test)**2, axis=1))
    
    # X_test와 가까운 K개의 이웃의 클래스와 그 이웃과의 거리를 리턴한다.
    def obtain_k_nearest_neighbor(self, X_test):
        # X_test와 X_train들 사이의 거리를 구한다. 
        # X_train은 클래스 생성자에서 받고있고, (# of train data, 4)의 형태를 갖는다.
        # X_test는 (4, )의 형태를 갖는다.
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
        
    def obtain_majority_vote(self, X_test):
        # X_test 와 가장 가까운 k개의 이웃을 구한다.
        neighbors, distance = self.obtain_k_nearest_neighbor(X_test)

        # neighbors에는 X_test와 가까운 K개의 이웃의 클래스가 들어있다.
        # 이중 가장 많이 나온 클래스를 X_test의 클래스로 예측한다.

        # 가장 많이 나온 클래스를 구하기 위해 vote라는 넘파이 배열을 생성한다.
        # vote의 index는 class를 뜻하고, value는 neighbors에 해당 class가 나온 횟수이다.
        vote = np.zeros(neighbors.max() + 1)
        for i in range(neighbors.size):
            vote[neighbors[i]] += 1

        # vote에서 값이 가장 큰 index(=class) 를 리턴한다.
        return np.argmax(vote)
        
    def obtain_weighted_majority_vote(self, X_test):
        # X_test 와 가장 가까운 k개의 이웃을 구한다.
        neighbors, distance = self.obtain_k_nearest_neighbor(X_test)
        
        # neighbors에는 X_test와 가까운 K개의 이웃의 클래스가 들어있다.
        # 이중 가장 많이 나온 클래스를 X_test의 클래스로 예측한다.

        # 넘파이 배열 weighted_vote를 생성한다.
        # obtain_majority_vote 메소드의 vote 배열과 다른점은, 
        # 각 이웃까지의 거리를 사용해 가중치를 적용했다는 점 이다.
        weighted_vote = np.zeros(neighbors.max() + 1)
        for i in range(neighbors.size):
            # 여기서는 가중치를 거리에 반비례하도록 구현했다.
            weighted_vote[neighbors[i]] += 1 / distance[i]
        
        return np.argmax(weighted_vote)
