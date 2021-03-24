# 다차원 배열의 계산
import numpy as np

def _print(x):

    print("\nprint(x): {}".format(x))
    print("np.ndim(x): {}".format(np.ndim(x)))
    print("x.shape: {}".format(x.shape))
    print("x.shape[0]: {}".format(x.shape[0]))


A = np.array([1, 2, 3, 4])
_print(A)

B = np.array([[1, 2], [3, 4], [5, 6]])
_print(B)

C = np.array([[[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]]])
_print(C)


A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(np.dot(A, B))
