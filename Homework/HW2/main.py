# 부모 디렉토리에서 import할 수 있도록 설정
import sys, os
sys.path.append(os.pardir)

import numpy as np
import time
from KNN import KNN
# mnist data load할 수 있는 함수 import
from dataset.mnist import load_mnist

# Hand-craft function
def handCraft(x):
    # 시간 측정을 위한 타이머
    start = time.perf_counter()

    x_handCrafted = []
    for xi in x:
        temp = []
        x_reshaped = xi.reshape(28, 28)
        for i in range(28):
            count_ij = 0
            count_ji = 0
            for j in range(28):
                if (x_reshaped[i][j] != 0):
                    count_ij += 1
                if (x_reshaped[j][i] != 0):
                    count_ji += 1
            temp.append(count_ij)
            temp.append(count_ji)
        x_handCrafted.append(temp)
    
    print("# hand craft processing time : {}s".format(time.perf_counter() - start))
    return np.array(x_handCrafted)

# 테스트 함수 선언
def Test(K, X_train, y_train, X_test, y_test, y_name, sample):
    # 시간 측정을 위한 타이머
    start = time.perf_counter()
    # KNN 알고리즘을 사용하는 분류기 생성
    classifier = KNN(K, X_train, y_train)

    # weighted_majority_vote를 사용한 분류 결과 계산 및 출력
    # accuracy 계산을 위해 정답을 맞춘 횟수를 저장
    accurate_count = 0
    for i in sample:
        computed_class = classifier.obtain_weighted_majority_vote(X_test[i])
        print("{index} th data\tresult {result}\tlabel {label}".format(index=i, result=y_name[computed_class], label=y_name[y_test[i]]))
        if (computed_class == y_test[i]):
            accurate_count += 1
    
    end = time.perf_counter()
    print("accuracy = {}".format(accurate_count / sample.size))
    print("sample size: {sample_size}, K: {k}, performance time: {time}s".format(sample_size=sample.size, k=K, time=(end - start)))


# training data, test data
# flatten: 이미지를 1차원 배열로 읽음
# normalize: 0~1 실수로. 그렇지 않으면 0~255
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# load_mnist 함수의 리턴값은 전부 uint8 형태이다.
# 이 상태로 KNN 알고리즘을 실행하면 두 점 사이의 거리를 구하는 과정에서 underflow가 나게 된다.
# 따라서 실행상 계산의 편의를 위해 int32 타입으로 형변환 해준다.
(x_train, t_train), (x_test, t_test) = (x_train.astype(np.int32), t_train.astype(np.int32)), (x_test.astype(np.int32), t_test.astype(np.int32))

# test data 10,000개 중 일부를 랜덤하게 샘플링해서 사용
size = 100
sample = np.random.randint(0, t_test.shape[0], size)

# KNN의 3번째 파라미터(label의 이름)으로 다음의 리스트 사용
label_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


# 테스트
x_handCrafted_train = handCraft(x_train)
x_handCrafted_test = handCraft(x_test)
for k in [1, 3, 5, 10, 15, 20]:
    print("\n# Test : K = {}, input feature = original(784 features)".format(k))
    Test(k, x_train, t_train, x_test, t_test, label_name, sample)
    
    print("\n# Test : K = {}, input feature = hand crafted".format(k))
    Test(k, x_handCrafted_train, t_train, x_handCrafted_test, t_test, label_name, sample)
