
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 22:49:15 2018

@author: User
"""

import dicom
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
import math
import numpy as np
import time
#import cv2
import copy
from sklearn import preprocessing
import scipy.optimize as opt
#File=['IM1.dcm']
File=[]
start=time.time()
for i in range(120,121):
    File.append('IM'+str(i))
                                        
def convert2polar(x,y,image_size,center):
    r = np.sqrt((x-center)**2+(y-center)**2)
    theta = np.arctan2(y-center,x-center)* 180 / np.pi
    return [r,theta]

def image_feature(image, image_size):
    center = 256
    location_x = (np.array([np.arange(image_size).tolist() for i in range(image_size)])-center)/center
    location_y = (np.array([(i*np.ones(image_size)).tolist() for i in range(image_size)])-center)/center
    image = preprocessing.scale(np.array(image))
    #image_x = np.gradient(image)[1]
    #image_y = np.gradient(image)[0]
    #image_xx = np.gradient(image_x)[1]
    #image_yy = np.gradient(image_y)[0]
    #image_xy = np.gradient(image_x)[0]
    #curvature_image = preprocessing.scale((image_x**2*image_yy + image_xx*image_y**2 - 2*image_xy)/((image_x**2+image_y**2 + 1e-04)**(3/2)))
    #det = image_xx * image_yy - image_xy**2
    r = np.sqrt((location_x)**2+(location_y)**2) / (np.max(np.sqrt((location_x)**2+(location_y)**2)))
    theta = np.arctan2(location_y,location_x) / (np.max(np.arctan2(location_y,location_x)))
    return[location_x,location_y,image,r,theta]#,curvature_image]
    #return[location_x/512,location_y/512,image/(np.amax(image) - np.amin(image))]#,r/(np.amax(r) - np.amin(r))]
def Gaussian_Mixture_Model(feature_matrix,cluster): # x,y,I,Ix,Iy,Ixy,Ixx,Iyy,k,det,r,theta
    feature_0 = feature_matrix[0].flatten() #x
    feature_1 = feature_matrix[1].flatten() #y
    feature_2 = feature_matrix[2].flatten() #I
    feature_3 = feature_matrix[3].flatten() #r
    feature_4 = feature_matrix[4].flatten() #theta
    # feature_5 = feature_matrix[5].flatten()  #curvature_image
    # feature_6 = feature_matrix[6].flatten()
    # feature_7 = feature_matrix[7].flatten()
    # feature_8 = feature_matrix[8].flatten()
    # feature_9 = feature_matrix[9].flatten()
    # feature_10 = feature_matrix[10].flatten()
    # feature_11 = feature_matrix[11].flatten()
    X = [[np.absolute(feature_4[i]),feature_2[i],feature_3[i]] for i in range(0,len(feature_2))]#,feature_3[i],feature_4[i],feature_5[i],feature_6[i],feature_7[i],feature_8[i],feature_9[i],feature_10[i],feature_11[i]] for i in range(0,len(feature_0))]
    gmm_mean = np.transpose(GMM(n_components=cluster).fit(X).means_)
    
    #### fmin ####
    Y = solve_matrix_lsq(X,gmm_mean)
    #Y = np.matmul(np.linalg.pinv(gmm_mean),np.transpose(X))
    #labels = gmm.predict(X)
    return [gmm_mean,Y]

def solve_matrix_lsq(X,M): #solve X = M*Y
    Y = []
    for i in range(0,len(X)):
        tmp = opt.nnls(M,np.transpose(X[i]))
        Y.append(tmp[0])
    return np.transpose(Y)

def atalas(input):
    return([input[0:256,0:256],input[0:256,256:512],input[256:512,0:256],input[256:512,256:512]])
def main(file):
    ds=dicom.read_file('D:\\MRI_segmentation\\T1 3D\\SE12\\'+file)#+'.dcm')
    #ds=dicom.read_file(file)#+'.dcm')
    
    
    pixel_size=len(ds.pixel_array)
     
    ds_pixel = ds.pixel_array
    #phi = (initial_level_set(pixel_size,390,130,20)) #390 130 20

    image_feature_data = image_feature(ds_pixel,pixel_size)

    #####################
    # NUMBER OF CLUSTER #
    ##################### 
    CLUSTER = 3

    #polar_coordinate = convert2polar(image_feature_data[0],image_feature_data[1],pixel_size,int(pixel_size/2))
    chart_11 = []
    chart_12 = []
    chart_21 = []
    chart_22 = []
    for feature in image_feature_data:
        chart_11.append(atalas(feature)[0])
        chart_12.append(atalas(feature)[1])
        chart_21.append(atalas(feature)[2])
        chart_22.append(atalas(feature)[3])
    atalas_class = [chart_11,chart_12,chart_21,chart_22]              
    #mean = Gaussian_Mixture_Model(image_feature_data)[1]
    #X = Gaussian_Mixture_Model(image_feature_data)[1]
    label = np.ones((512,512))
    for chart_no in range(1):
        Y = Gaussian_Mixture_Model(image_feature_data,CLUSTER)[1]
        # location = Gaussian_Mixture_Model(atalas_class[chart_no],CLUSTER)[0]
        # print(location)
        print(Gaussian_Mixture_Model(image_feature_data,CLUSTER)[0])
        label_local = []
        
        for i in range(len(Y[0])):
            #print(Y[:,i])
            #print(np.where(np.amax(Y[:,i]))[0][0])
            #np.append(label,(np.where(np.amax(Y[:,i]))[0][0]))
            tmp = (Y[:,i]).tolist() 
            label_local.append(tmp.index(max(tmp)))#+chart_no*CLUSTER)
        #print(label)
    label_local = np.reshape(label_local,(512,512))  
    
    for i in range(CLUSTER*0,CLUSTER*1):
        image = ds_pixel*((label_local==i).astype(np.int))
        plt.imshow(image,cmap = plt.cm.bone)
        plt.suptitle("The label" + str(i))
        
        plt.imsave("Cluster" + str(i)+".png",image,cmap = plt.cm.bone,  dpi = 1500)
        plt.show() 
    #print(label[512*200+25])
    #segment = np.array(ds_pixel)*((1-(np.reshape((label == 8).astype(np.int),(512,512)))) + (1-(np.reshape((label == 9).astype(np.int),(512,512)))))
    #for i in range(10):
    #    segment = ((np.reshape((label == i).astype(np.int),(512,512))))
    #    plt.imshow(segment,cmap = plt.cm.bone)
    #    plt.suptitle("this is "+str(i))
    #    plt.show()    
    




start=time.time()

for i in File:
    main(i)
end=time.time()
print(end-start)