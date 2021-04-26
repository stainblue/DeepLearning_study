# 부모 디렉토리에서 import할 수 있도록 설정
import sys, os
sys.path.append(os.pardir)

import numpy as np
# mnist data load할 수 있는 함수 import
from dataset.mnist import load_mnist

# python image processing library
# python 버전 3.x 에서는 pillow package install해서 사용
from PIL import Image

# Hand-craft function
def handCraft(x):
    pass

# 784개의 input을 그대로 사용하여 분류 테스트
def test_original_input(K, X_train, y_train, X_test, y_test, y_name):
    print("# Test : input feature : original(784 features)")
    # KNN 알고리즘을 사용하는 분류기 생성
    classifier = KNN(K, X_train, y_train)

    # weighted_majority_vote를 사용한 분류 결과 계산 및 출력
    for i in range(X_test.shape[0]):
        computed_class = classifier.obtain_weighted_majority_vote(X_test[i])

def test_handCrafted_input(K, X_train, y_train, X_test, y_test, y_name):
    pass

# training data, test data
# flatten: 이미지를 1차원 배열로 읽음
# normalize: 0~1 실수로. 그렇지 않으면 0~255
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# test data 10,000개 중 일부를 랜덤하게 샘플링해서 사용
size = 100
sample = np.random.randint(0, t_test.shape[0], size)
for i in sample:


# KNN의 3번째 파라미터(label의 이름)으로 다음의 리스트 사용
label_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
