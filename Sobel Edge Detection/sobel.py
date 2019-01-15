import numpy as np
import cv2
import math

img = cv2.imread('task1.png',0)  
cv2.imshow('detected',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
Gy = [[1,0,-1],[2,0,-2],[1,0,-1]]
Gx = [[1,2,1],[0,0,0],[-1,-2,-1]]
arr1 = np.array(Gx)
arr2 = np.array(Gy)

def sobel_x():
    print('PREPARAING SOBEL-X')
    img_pad = np.pad(img, 1, mode='constant')
    lengthimg = np.size(img)
    lengthker = np.size(arr1)
    img_h = img_pad.shape[0]
    img_w = img_pad.shape[1]
    arr1_h = arr1.shape[0]
    arr1_w = arr1.shape[1]  
    h = arr1_h//2
    w = arr1_w//2   
    img_convx = np.zeros(img_pad.shape)   
    for i in range(h,img_h-h):
        for j in range(w,img_w-w):
            sum = 0
            
            for x in range(arr1_h):
                for y in range(arr1_w):
                    sum = sum + (arr1[x][y] * img_pad[i-h+x][j-w+y])
            img_convx[i][j] = sum
            
    cv2.imwrite('Convolved 2D-X.jpg',img_convx)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    p1=cv2.imread('Convolved 2D-X.jpg',0)
    cv2.imshow('Convolved 2D-X.jpg',p1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img_convx

def sobel_y():
    print('PREPARING SOBEL Y')
    img_pad = np.pad(img, 1, mode='constant')
    lengthimg = np.size(img)
    lengthker = np.size(arr1)
    img_h = img_pad.shape[0]
    img_w = img_pad.shape[1]
    arr2_h = arr2.shape[0]
    arr2_w = arr2.shape[1]  
    h = arr2_h//2
    w = arr2_w//2   
    img_convy = np.zeros(img_pad.shape)   
    for i in range(h,img_h-h):
        for j in range(w,img_w-w):
            sum = 0
            
            for x in range(arr2_h):
                for y in range(arr2_w):
                    sum = sum + (arr2[x][y] * img_pad[i-h+x][j-w+y])
            img_convy[i][j] = sum
            
    cv2.imwrite('Convolved 2D-Y.jpg',img_convy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    p1=cv2.imread('Convolved 2D-Y.jpg',0)
    cv2.imshow('Convolved 2D-Y.jpg',p1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img_convy

def main_func():
    x = sobel_x()
    y = sobel_y()
main_func()