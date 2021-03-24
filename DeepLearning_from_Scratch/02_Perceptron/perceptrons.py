# implement perceptron

# initial version
print("# inital version")
def AND(x1, x2):
    w1, w2, theta = 1, 1, 1
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

print(AND(0, 0))
print(AND(1, 0))
print(AND(0, 1))
print(AND(1, 1))

# introduce bias
print("\n# introduce bias")
import numpy as np
x = np.array([0, 1])    # input
w = np.array([1, 1])    # weight
b = -1                  # bias
print(w * x)
print(np.sum(w*x))
print(np.sum(w*x) + b)

# apply bias
print("\n# apply bias")
print("## AND")
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([1, 1])
    b = -1
    tmp = np.sum(x * w) + b
    if tmp <= 0:
        return 0
    else:
        return 1

print(AND(0, 0))
print(AND(1, 0))
print(AND(0, 1))
print(AND(1, 1))

print("## NAND")
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 1
    tmp = np.sum(x * w) + b
    if tmp <= 0:
        return 0
    else:
        return 1

print(NAND(0, 0))
print(NAND(1, 0))
print(NAND(0, 1))
print(NAND(1, 1))

print("## OR")
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([1, 1])
    b = 0
    tmp = np.sum(x * w) + b
    if tmp <= 0:
        return 0
    else:
        return 1

print(OR(0, 0))
print(OR(1, 0))
print(OR(0, 1))
print(OR(1, 1))


# multi-layer perceptron
print("\nmulti-layer perceptron")
print("## XOR")
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
print(XOR(0, 0))
print(XOR(1, 0))
print(XOR(0, 1))
print(XOR(1, 1))

