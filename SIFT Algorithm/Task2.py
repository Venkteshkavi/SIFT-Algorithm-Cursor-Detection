import cv2
import numpy as np
import math
#from tqdm import tqdm
#from time import sleep
pi = math.pi
def image_reading():
    img = cv2.imread("task2.jpg",0)
    #print(img.shape)
    cv2.imshow('Det',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img

#Initializing dictionary containing 25 gaussian matrices to zero
def gauss_initialization():
    gauss_list = {}
    gauss = np.zeros((7,7),dtype=float)
    gauss_list = {}
    for i in range(0,25):
        gauss_list["gauss"+str(i)] = gauss
    return gauss_list


# Preparation of 25 gaussian matrices with 25 Sigma Values
def gaussian():
    print('SMOOTHENING IN PROGRESS')
    gauss_list = gauss_initialization()
    a = [-3,-2,0,1,2,3]
    gauss_matrix = np.zeros((7,7), dtype= 'float32')
    sig = [1/math.sqrt(2), 1, math.sqrt(2), 2, 2*math.sqrt(2), math.sqrt(2), 2, 2*math.sqrt(2), 4, 4*math.sqrt(2), 2*math.sqrt(2), 2, 2*math.sqrt(2), 4, 4*math.sqrt(2), 2*math.sqrt(2), 4, 4*math.sqrt(2), 8, 8*math.sqrt(2), 4*math.sqrt(2), 8, 8*math.sqrt(2), 16, 16*math.sqrt(2)]
    sig_len = len(sig)
    a = 0
    while(a<25):
        #print(a)
        #print('')
        gauss_matrix = np.zeros((7,7), dtype= 'float32')
        for i in (range(3,-4,-1)):
            for j in range(-3,4):
                gauss_matrix[3-i][j+3] = (1/(2*pi*sig[a]**2)) * math.exp(-((i**2) + (j**2)) / (2*sig[a]**2))        
        
        gauss_list["gauss"+str(a)] = gauss_matrix        
        #Incrementing sigma value to go till 25 values
        a = a+1
    #aprint(gauss_list)
    #print(gauss_list["gauss0"])
    return gauss_list



def finalplotting(kpy,resimg1,e,scaling_factor):
    resimg1 = cv2.imread('task2.jpg')
    print('PLOTTING RESULTS')
    for i in range(len(kpy)):
        index_x = kpy[i][0]
        index_y = kpy[i][1]
        resimg1[index_x * scaling_factor][index_y * scaling_factor] = 255
    print('GENERATING O/P IMAGE')
    print('GEARING UP FOR FINAL PRINT')
    #print(resimg1)
    cv2.imwrite('SIFT image' + str(e) + '.jpg',resimg1)
    cv2.imshow('Result',resimg1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    e= e+1



#Storing gauss matrix for each sigma in dictionary with keys(gauss1, gauss2). Here we also convert int a to string to concatenate to get the key
def octaveprep1():
    print('GENERATING OCTAVE 1')
    resimg1 = image_reading()  
    gauss_list = gaussian()
    #print(imgr)
    img_pad = {}
    img_pad1 = np.pad(resimg1, 3, mode='constant')
    img_pad1 = np.asarray(resimg1, dtype= 'float32')
    #print(img_pad1)
    img_h = resimg1.shape[0]
    img_w = resimg1.shape[1]

    #transpose = [[test[j][i] for j in range(len(test))] for i in range(len(test[0]))]

    gauss_h = 7
    #print(gauss_h)
    gauss_w = 7
    #print(gauss_w)
    h = gauss_h//2
    w = gauss_w//2
    a=0
    b=0
    octave1 = {}
    a=0
    while (a<5):
        img_conv = np.zeros(resimg1.shape, dtype= 'float32')
        img_conv= np.asarray(img_conv,dtype='float32')
        for p in range(h,img_h-h):
            for q in range(w,img_w-w):
                sum = 0        
                sum = np.asarray(sum, dtype= 'float32')
                for r in range(gauss_h):
                    for s in range(gauss_w):
                        sum = sum + (gauss_list["gauss" +str(a)][r][s] * img_pad1[p-h+r][q-w+s])
                        img_conv[p-h][q-w] = sum
        a = a+1
        octave1["blurred" + str(b)] = img_conv
        b = b+1     
    print('OCTAVED GENERATED')
    #print(octave1)
    return octave1      
#    print(img_conv)
#    cv2.imwrite('Original Gray Image.jpg',imgr)
#    cv2.imwrite('convolved output.jpg',img_conv)
#    cv2.imshow('Det',img_conv)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

def octaveprep2():
    resimg1 = image_reading()
    resimg2 = resimg1[::2]
    resimg2 = resimg2[:,1::2]
    #cv2.imshow('Resized2',resimg2)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    gauss_list = gaussian()
    #print(imgr)
    #for x in range(3):
    #    img_pad["img_pad"+str(x)] = np.pad(resimg + str(x+1), 3, mode='constant')
    img_pad2 = np.pad(resimg2, 3, mode='constant')
    img_pad2 = np.asarray(resimg2, dtype= 'float32')
    #print(img_pad2)
    #print(img_pad)
    img_h = resimg2.shape[0]
    img_w = resimg2.shape[1]
    #print(img_h)  cv2.waitKey(0)
   
    #print(img_w)
    ##print(gauss_list["gauss1"])
    #test = gauss_list["gauss0"]
    #test = np.asarray(test, dtype='float32')
    #print(test)
    #transpose = [[test[j][i] for j in range(len(test))] for i in range(len(test[0]))]
    #print(transpose)
    gauss_h = 7
    #print(gauss_h)
    gauss_w = 7
    #print(gauss_w)
    h = gauss_h//2
    w = gauss_w//2
    a=0
    b=0
    octave2 = {}
    a=5
    while (a<10):
        img_conv = np.zeros(resimg1.shape, dtype= 'float32')
        img_conv= np.asarray(img_conv,dtype='float32')
        for p in range(h,img_h-h):
            for q in range(w,img_w-w):
                sum = 0        
                sum = np.asarray(sum, dtype= 'float32')
                for r in range(gauss_h):
                    for s in range(gauss_w):
                        sum = sum + (gauss_list["gauss" +str(a)][r][s] * img_pad2[p-h+r][q-w+s])
                        img_conv[p-h][q-w] = sum
        a = a+1
        octave2["blurred" + str(b)] = img_conv
        b = b+1            
    print('GENERATING OCTAVE 2')
    #print(octave2)
    #print('***************************************NEXT OCTAVE STARTS****************************************************************')
    return octave2

def octaveprep3():
    resimg1 = image_reading()
    resimg3 = resimg1[::4]
    resimg3 = resimg3[:,1::4]
    #cv2.imshow('Resized3',resimg3)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    gauss_list = gaussian()
    #print(imgr)
    #for x in range(3):
    #    img_pad["img_pad"+str(x)] = np.pad(resimg + str(x+1), 3, mode='constant')
    img_pad3 = np.pad(resimg3, 3, mode='constant')
    img_pad3 = np.asarray(resimg3, dtype= 'float32')
    #print(img_pad3)
    #print(img_pad)
    img_h = resimg3.shape[0]
    img_w = resimg3.shape[1]
    #print(img_h)  cv2.waitKey(0)
   
    #print(img_w)
    ##print(gauss_list["gauss1"])
    #test = gauss_list["gauss0"]
    #test = np.asarray(test, dtype='float32')
    #print(test)
    #transpose = [[test[j][i] for j in range(len(test))] for i in range(len(test[0]))]
    #print(transpose)
    gauss_h = 7
    #print(gauss_h)
    gauss_w = 7
    #print(gauss_w)
    h = gauss_h//2
    w = gauss_w//2  
    a=0
    b=0
    octave3 = {}
    a=10
    while (a<15):
        img_conv = np.zeros(resimg3.shape, dtype= 'float32')
        img_conv= np.asarray(img_conv,dtype='float32')
        for p in range(h,img_h-h):
            for q in range(w,img_w-w):
                sum = 0        
                sum = np.asarray(sum, dtype= 'float32')
                for r in range(gauss_h):
                    for s in range(gauss_w):
                        sum = sum + (gauss_list["gauss" +str(a)][r][s] * img_pad3[p-h+r][q-w+s])
                        img_conv[p-h][q-w] = sum
        a = a+1
        octave3["blurred" + str(b)] = img_conv
        b = b+1            
    print('GENERATING OCTAVE 3')
    #print(octave3)
    #print('***************************************NEXT OCTAVE STARTS****************************************************************')
    return octave3

def octaveprep4():
    resimg1 = image_reading()
    
    resimg4 = resimg1[::8]
    resimg4 = resimg4[:,1::8]
    #cv2.imshow('Resized4',resimg4)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    gauss_list = gaussian()
    #print(imgr)
    #for x in range(3):
    #    img_pad["img_pad"+str(x)] = np.pad(resimg + str(x+1), 3, mode='constant')
    img_pad4 = np.pad(resimg4, 3, mode='constant')
    img_pad4 = np.asarray(resimg4, dtype= 'float32')
    #print(img_pad)
    img_h = resimg4.shape[0]
    img_w = resimg4.shape[1]
    #print(img_h)  cv2.waitKey(0)
   
    #print(img_w)
    ##print(gauss_list["gauss1"])
    #test = gauss_list["gauss0"]
    #test = np.asarray(test, dtype='float32')
    #print(test)
    #transpose = [[test[j][i] for j in range(len(test))] for i in range(len(test[0]))]
    #print(transpose)
    gauss_h = 7
    #print(gauss_h)
    gauss_w = 7
    #print(gauss_w)
    h = gauss_h//2
    w = gauss_w//2
    a=0
    b=0
    octave4 = {}
    a=15
    while (a<20):
        img_conv = np.zeros(resimg1.shape, dtype= 'float32')
        img_conv= np.asarray(img_conv,dtype='float32')
        for p in range(h,img_h-h):
            for q in range(w,img_w-w):
                sum = 0        
                sum = np.asarray(sum, dtype= 'float32')
                for r in range(gauss_h):
                    for s in range(gauss_w):
                        sum = sum + (gauss_list["gauss" +str(a)][r][s] * img_pad4[p-h+r][q-w+s])
                        img_conv[p-h][q-w] = sum
        a = a+1
        octave4["blurred" + str(b)] = img_conv
        b = b+1            
    print('GENERATING OCTAVE 4')
    #print(octave4)
    print('')
    return octave4

def dogprep1():
    scaling_factor = 1
    print('')
    print('ENTERING DOG PREPARATION 1')
    orgimg = cv2.imread('task2.jpg')
    resimg1 = image_reading()
    octave1 = octaveprep1()
    dog1 = {}
    sub = np.zeros(resimg1.shape, dtype='float32')
    sub = np.asarray(sub, dtype='float32')
    for i in range(4):
         sub = octave1["blurred" + str(i)] - octave1["blurred" + str(i+1)] 
         #print('___SUBTRACTED MATRIX___')
         #print('')
         #print(sub)
         #print('')
         dog1["grad"+str(i)] = sub
    
    #print(dog1)
    keypoints1 = keypointprep_1(dog1)
    keypoints2 = keypointprep_2(dog1)
    finalplotting(keypoints1,orgimg,1,scaling_factor)
    finalplotting(keypoints2,orgimg,2,scaling_factor)


def dogprep2():
    scaling_factor = 2
    print('')
    print('ENTERING DOG PREPARATION 2')
    orgimg = cv2.imread('task2.jpg')
    orgimg_resized1 = orgimg[::2]
    orgimg_resized1 = orgimg_resized1[:,1::2]
    cv2.imwrite("Rescaled Image 2.jpg",orgimg_resized1)
    resimg1 = image_reading()
    resimg2 = resimg1[::2]
    resimg2 = resimg2[:,1::2]
    octave2 = octaveprep2()
    dog2 = {}
    sub = np.zeros(resimg1.shape, dtype='float32')
    sub = np.asarray(sub, dtype='float32')
    for i in range(4):
         sub = octave2["blurred" + str(i)] - octave2["blurred" + str(i+1)] 
         #print('___SUBTRACTED MATRIX___')
         #print('')
         #print(sub)
         #print('')
         dog2["grad"+str(i)] = sub
    cv2.imwrite("DOG2 Image 1.jpg",dog2["grad0"])
    cv2.imwrite("DOG2 Image 2.jpg",dog2["grad1"])
    cv2.imwrite("DOG2 Image 3.jpg",dog2["grad2"])
    cv2.imwrite("DOG2 Image 4.jpg",dog2["grad3"])
    keypoints1 = keypointprep_1(dog2)
    keypoints2 = keypointprep_2(dog2)
    #print(keypoints1)
    #print(keypoints2)
    finalplotting(keypoints1,orgimg,3,scaling_factor)
    finalplotting(keypoints2,orgimg,4,scaling_factor)

def dogprep3():
    scaling_factor = 4
    print('')
    print('ENTERING DOG PREP 3')
    orgimg = cv2.imread('task2.jpg')
    orgimg_resized2 = orgimg[::4]
    orgimg_resized2 = orgimg_resized2[:,1::4]
    cv2.imwrite("Rescaled Image 3.jpg",orgimg_resized2)
    resimg1 = image_reading()
    resimg3 = resimg1[::4]
    resimg3 = resimg3[:,1::4]
    octave3 = octaveprep3()
    dog3 = {}
    sub = np.zeros(resimg3.shape, dtype='float32')
    sub = np.asarray(sub, dtype='float32')
    for i in range(4):
         sub = abs(octave3["blurred" + str(i)] - octave3["blurred" + str(i+1)])
         #print('___SUBTRACTED MATRIX___')
         #print('')
         #print(sub)
         #print('')
         dog3["grad"+str(i)] = sub
    cv2.imwrite("DOG3 Image 1.jpg",dog3["grad0"])
    cv2.imwrite("DOG3 Image 2.jpg",dog3["grad1"])
    cv2.imwrite("DOG3 Image 3.jpg",dog3["grad2"])
    cv2.imwrite("DOG3 Image 4.jpg",dog3["grad3"])
    #print(dog3)
    keypoints1 = keypointprep_1(dog3)
    keypoints2 = keypointprep_2(dog3)
    finalplotting(keypoints1,orgimg,5,scaling_factor)
    finalplotting(keypoints2,orgimg,6,scaling_factor)

def dogprep4():
    scaling_factor = 8
    print('')
    print('ENTERING DOG PREP 4')
    orgimg = cv2.imread('task2.jpg')
    orgimg_resized3 = orgimg[::8]
    orgimg_resized3 = orgimg_resized3[:,1::8]
    resimg1 = image_reading()
    resimg4 = resimg1[::8]
    resimg4 = resimg4[:,1::8]
    octave4 = octaveprep4()
    dog4 = {}
    sub = np.zeros(resimg1.shape, dtype='float32')
    sub = np.asarray(sub, dtype='float32')
    for i in range(4):
        sub = abs(octave4["blurred" + str(i)] - octave4["blurred" + str(i+1)])
        #print('___SUBTRACTED MATRIX___')
        #print('')
        #print(sub)
        #print('')
        dog4["grad"+str(i)] = sub
    keypoints1 = keypointprep_1(dog4)
    keypoints2 = keypointprep_2(dog4)
    finalplotting(keypoints1,orgimg,7,scaling_factor)
    finalplotting(keypoints2,orgimg,8,scaling_factor)

def keypointprep_1(dog_ret):
    print('PREPARAING KEY POINTS 1')
    allmatrix_points = list()
    keypoints1 = list()
    h = 1
    w= 1
    gauss_h = 3
    gauss_w = 3
    keymatrix_1 = dog_ret["grad0"]
    keymatrix_2 = dog_ret["grad1"]
    keymatrix_3 = dog_ret["grad2"]
    keymatrix_4 = dog_ret["grad3"]
    keymatrix_row = keymatrix_2.shape[0]
    keymatrix_col = keymatrix_2.shape[1]
    for p in range(keymatrix_row-3):
        for q in range(keymatrix_col-3):
            mid_pixel = keymatrix_2[p+1][q+1]
            allmatrix_points = list()
            for r in range(gauss_h):
                for s in range(gauss_w):
                    allmatrix_points.append(keymatrix_1[p+r][q+s])
                    allmatrix_points.append(keymatrix_2[p+r][q+s])
                    allmatrix_points.append(keymatrix_3[p+r][q+s])
                    if(mid_pixel < min(allmatrix_points) or mid_pixel > max(allmatrix_points)):
                        #print(mid_pixel)
                        keypoints1.append((p+1,q+1))

    #print(keypoints1)
    return keypoints1
def keypointprep_2(dog_ret):
    print('PREPARAING KEY POINTS 2')
    allmatrix_points = list()
    keypoints2 = list()
    h = 1
    w= 1
    gauss_h = 3
    gauss_w = 3
    keymatrix_1 = dog_ret["grad0"]
    keymatrix_2 = dog_ret["grad1"]
    keymatrix_3 = dog_ret["grad2"]
    keymatrix_4 = dog_ret["grad3"]
    keymatrix_row = keymatrix_2.shape[0]
    keymatrix_col = keymatrix_2.shape[1]
    for p in range(keymatrix_row-3):
        for q in range(keymatrix_col-3):
            mid_pixel = keymatrix_3[p+1][q+1]
            allmatrix_points = list()
            for r in range(gauss_h):
                for s in range(gauss_w):
                    allmatrix_points.append(keymatrix_2[p+r][q+s])
                    allmatrix_points.append(keymatrix_3[p+r][q+s])
                    allmatrix_points.append(keymatrix_4[p+r][q+s])
                    if(mid_pixel < min(allmatrix_points) or mid_pixel > max(allmatrix_points)):
                        #print(mid_pixel)
                        keypoints2.append((p+1,q+1))
          
    #print(keypoints2)
    return keypoints2

#dogprep1()
#dogprep2()
dogprep3()
#dogprep4()