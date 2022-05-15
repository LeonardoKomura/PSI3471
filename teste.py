import math
import numpy as np
A = np.ones((3, 1))
print(A)

B = A + [[1], [2], [3]]
print(B)

AF = A.flatten(order='C')
BF = B.flatten(order='C')
print(AF)
print(BF)
C = np.dot(AF, BF)
print(C)