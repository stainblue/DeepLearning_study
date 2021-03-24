# numpy practice

import numpy as np

def print_op(x, y):
    print("x : {value}, type : {type}".format(value = x, type = type(x)))
    print("y : {value}, type : {type}".format(value = y, type = type(y)))

    print("x + y : {}".format(x + y))
    print("x - y : {}".format(x - y))
    print("x * y : {}".format(x * y))
    print("x / y : {}\n".format(x / y))


x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
print_op(x, y)  # element-wise

# y = np.array([2.0, 4.0, 6.0, 8.0])
# print_op(x, y)
# 
# Traceback (most recent call last):
#   File ".\numpy_practice.py", line 17, in <module>
#     print_op(x, y)
#   File ".\numpy_practice.py", line 6, in print_op
#     print("x + y : {}".format(x + y))
# ValueError: operands could not be broadcast together with shapes (3,) (4,)

y = 2.0
print_op(x, y)  # broadcast


# numpy N-dimension
A = np.array([[1, 2], [3, 4]])
print(A)
print(A.shape)
print(A.dtype)

B = np.array([[3, 1], [1, 6]])
print_op(A, B)

# broadcast
A = np.array([[1, 2], [3, 4]])
B = np.array([10, 20])
print_op(A, B)

# 원소 접근
X = np.array([[1, 2], [3, 4], [5, 6]])
print(X)
print(X[0])
print(X[0][1])
for row in X:
    print(row)
    for element in row:
        print(element)

X = X.flatten()
print(X)
print(X[np.array([0, 2, 4])])
print(X > 3)
print(X[X > 3])