import math
import time
import numpy as np
from sklearn import neighbors
from sklearn import tree

def le(nomearq):
    with open(nomearq,"r") as f:
        linhas=f.readlines()
    linha0=linhas[0].split()
    nl=int(linha0[0]); nc=int(linha0[1])
    a=np.empty((nl,nc),dtype=np.float32)
    for l in range(nl):
        linha=linhas[l+1].split()
        for c in range(nc):
            a[l,c]=np.float32(linha[c])
    return a

ax=le("ax.txt"); ay=le("ay.txt")
qx=le("qx.txt"); qy=le("qy.txt")

#Árvore de decisões
arvore = tree.DecisionTreeClassifier()
arvore = arvore.fit(ax, ay)
qp_tree = arvore.predict(qx)
#print("qp: ", qp_tree)
erros=0
start_time = time.time()
for i in range(qy.size):
    if qp_tree[i]!=qy[i]: erros = erros+1
taxa = 100.0*erros/qy.size
print("Arvore de decisoes")
print("Tempo de execucao: %s segundos" % (time.time() - start_time))
print("Número de erros: ", erros)
print("Taxa de erros (%): ", taxa)

#Vizinho mais próximo
init_time = time.time()
vizinho = neighbors.KNeighborsClassifier(n_neighbors=1, weights="uniform")
vizinho.fit(ax,ay.ravel())
qp_nn = vizinho.predict(qx)
#print("qp: ", qp_nn)
erros=0
for i in range(qy.size):
    if qp_nn[i]!=qy[i]: erros = erros+1
taxa = 100.0*erros/qy.size
print("\nVizinho mais proximo")
print("Tempo de execucao: %s segundos" % (time.time() - init_time))
print("Número de erros: ", erros)
print("Taxa de erros (%): ", taxa)