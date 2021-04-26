import sys, os
sys.path.append(os.pardir)
# 부모 디렉토리에서 import할 수 있도록 설정

import numpy as np
from dataset.mnist import load_mnist
# mnist data load할 수 있는 함수 import

from PIL import Image
# python image processing library
# python 버전 3.x 에서는 pillow package install해서 사용

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
# training data, test data
# flatten: 이미지를 1차원 배열로 읽음
# normalize: 0~1 실수로. 그렇지 않으면 0~255
image = x_train[0]
label = t_train[0]
# 첫번째 데이터
print(label)
print(image.shape)

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
# image를 unsigned int로
image = image.reshape(28,28)
# 1차원 —> 2차원 (28x28)
print(image.shape)
img_show(image)
