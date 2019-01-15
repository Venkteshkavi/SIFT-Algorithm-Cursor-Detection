import cv2
import numpy as np
import imutils
#Reading template and Original Image


def template_matching(orgimg,m,colimg,template_using,l):
    temp = template_using
    print('GENERATING IMAGE'+ str(l) + str(m))
    img = orgimg
    img_blur = cv2.GaussianBlur(img,(3,3),1)
    img_laplacian = cv2.Laplacian(img_blur,cv2.CV_32F)
    max_value = 0
    scale = np.linspace(0.4,1.0,30)[::-1]
    w,h,coordinates = 0,0,0
    for i in range(len(scale)):
        rtemp = temp
        resized_template = imutils.resize(rtemp, width= int(rtemp.shape[1] * scale[i]))
        temp_blur = cv2.GaussianBlur(resized_template,(3,3),1)
        temp_laplacian = cv2.Laplacian(temp_blur,cv2.CV_32F)
        res = cv2.matchTemplate(img_laplacian, temp_laplacian, cv2.TM_CCOEFF_NORMED)
        values_filtered = cv2.minMaxLoc(res)
        if(values_filtered[1] > max_value):
            max_value = values_filtered[1]
            coordinates = values_filtered[3]
            w, h = resized_template.shape[::-1]
    cv2.rectangle(colimg, coordinates, (coordinates[0]+w, coordinates[1]+h), (0,0,255), 2)  
    cv2.imwrite('Templated Matched Image' + str(l) + str(m) + '.jpg',colimg)
    cv2.imshow('Result',colimg) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

 def image_reading():
     print('MATCHING FOR ORIGINAL TEMPLATE AND NORMAL IMAGES')
     #  Positive bg images iteration
     for z in range(16):
         try:
             pos_img = cv2.imread('pos_' + str(z+1) + '.jpg',0)
             pos_img_colored = cv2.imread('pos_' + str(z+1) + '.jpg')
             template_using = cv2.imread('template.png',0)
             template_matching(pos_img,z,pos_img_colored,template_using,0)
         except:
             pass
     #Negateive images iteration
     for j in range(10):
         try:
             neg_img = cv2.imread('neg_' + str(j+1) + '.jpg',0)
             neg_img_colored = cv2.imread('neg_' + str(j+1) + '.jpg')
             template_using = cv2.imread('template.png',0)
             template_matching(neg_img,j,neg_img_colored,template_using,1)
         except:
             pass
 image_reading()
