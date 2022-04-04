import math
import numpy as np

ax = np.array([46, 120, 165, 51, 110, 173])
ay = np.array(['B', 'C', 'A', 'B', 'C', 'A'])
qx = np.array([60, 168, 105])
qy = np.array(['B', 'A', 'C'])
qp = np.array(['F', 'F', 'F'])

menorAy = 'F'
iq = 0
ia = 0

for iq in range(qx.size):
    menorDist = max(ax)
    for ia in range(ax.size):
        dist = abs(qx[iq] - ax[ia])
        if dist<menorDist:
            menorDist = dist
            menorAy = ay[ia]
    qp[iq] = menorAy
        
print("Classificao obtida: ")
print(qp)
print("Classificacao correta: ")
print(qy)
