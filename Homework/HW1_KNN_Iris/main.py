import numpy as np
from sklearn.datasets import load_iris
from KNN import KNN

# 테스트를 진행하는 함수
def Test(K, X_train, y_train, X_test, y_test, y_name):
    # KNN 알고리즘을 사용하는 분류기 생성
    # train data를 생성자의 파라미터로 넣어줌
    classifier = KNN(K, X_train, y_train)

    # majority_vote를 사용한 분류 결과 계산 및 출력
    print("# obtain_majority_vote")
    for i in range(X_test.shape[0]):
        computed_class = classifier.obtain_majority_vote(X_test[i])
        print("Test Data Index: {test_data_index}, Computed class: {computed_class}, True class: {true_class}".format(test_data_index=i, computed_class=y_name[computed_class], true_class=y_name[y_test[i]]))

    # weighted_majority_vote를 사용한 분류 결과 계산 및 출력
    print("# obtain_weighted_majority_vote")
    for i in range(X_test.shape[0]):
        computed_class = classifier.obtain_weighted_majority_vote(X_test[i])
        print("Test Data Index: {test_data_index}, Computed class: {computed_class}, True class: {true_class}".format(test_data_index=i, computed_class=y_name[computed_class], true_class=y_name[y_test[i]]))
    

# iris 데이터를 불러온다.
iris = load_iris()

# X (iris 데이터), y (iris 클래스), y_name(iris 클래스에 대한 영문 이름 (확인용))
X = iris.data       # iris data input
y = iris.target     # iris target (label)
y_name = iris.target_names # iris target name

# 총 150개의 데이터를 140개의 train 데이터와, 10개의 test 데이터로 나눈다.
X_train = [] # train data가 저장될 배열
y_train = [] # train target이 저장될 배열
X_test = []  # test data가 저장될 배열
y_test = []  # test target이 저장될 배열

# Every 15-th data 를 Test data로 사용하기 위해 반복문을 돌린다.
# X.shape = (150, 4) 이므로 X.shape[0] = 150
# 즉 0 <= i < 150
for i in range(X.shape[0]):
    if (i + 1) % 15 == 0:
        # i = 14, 29, ...
        X_test.append(X[i])
        y_test.append(y[i])
    else:
        X_train.append(X[i])
        y_train.append(y[i])

# X_train, y_train, X_test, y_test 를 np.array로 바꿔준다.
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# k = 3, 5, 10 을 대입하여 테스트 진행
for k in [3, 5, 10]:
    print("\nK = {}".format(k))
    Test(k, X_train, y_train, X_test, y_test, y_name)