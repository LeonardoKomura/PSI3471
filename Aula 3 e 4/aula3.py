import cv2
img = cv2.imread('mickey.bmp', 0)

for i in range(1, img.shape[0]-1):
    for j in range(1, img.shape[1]-1):
        if (img[i,j-1]==0 and img[i,j+1]==0) and img[i,j]==255:
            #print("bit-1 = ", img[i,j-1], "bit = ", img[i,j], "bit+1 = ", img[i,j+1])
            img[i,j] = 0
            
cv2.imwrite('mickey_fix.png', img)
