# 계단함수 구현하기

import numpy as np

# def step_function(x):
#     if x > 0:
#         return 1
#     else:
#         return 0

# v = np.array([-1, 0, 1])
# print(step_function(v))
#ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

# numpy array 지원
def step_function(x):
    y = x > 0
    return y.astype(np.int)

v = np.array([-1, 0, 1])
print(step_function(v))

import matplotlib.pylab as plt

def step_function(x):
    return np.array(x > 0, dtype=np.int)

print(step_function(v))

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()


