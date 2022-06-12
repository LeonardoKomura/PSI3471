import cv2
import queue

def pintaAzul(a,li,ci):
    b = a.copy()
    q = queue.Queue()
    q.put(li) #1
    q.put(ci) #1
    while not q.empty(): #2
        l = q.get() #3
        c=q.get() #3
        if all(b[l,c,:]==[255,255,255]): #4
            b[l,c]=[255,0,0] #5
            q.put(l-1); q.put(c) #6-acima
            q.put(l+1); q.put(c) #6-abaixo
            q.put(l); q.put(c+1) #6-direita
            q.put(l); q.put(c-1) #6-esquerda
    return b;

a = cv2.imread('mickey_reduz.bmp',cv2.IMREAD_COLOR)
b = pintaAzul(a,159,165)
cv2.imwrite('pintaaz_py.png',b)
