import math
import numpy as np
#A = np.array([1, 2, 3, 4, 5])
#A1 = np.reshape(A, [-1,1]) 
#At = np.transpose(A)
#print(A)
#print(A1)
#B = np.array([1, 1, 1, 1, 1])
#B1 = np.reshape(B, [-1,1]) 
#print(B1)
#C = np.add(A1,B1)
#print(C)
#D = A1+1
#print(D)

a = np.array([[1,2,3],[2,2,2],[5,5,5]])
print(a)
for i in range(3):
    b = a[i:(i+1), 0:3]
    print(b)