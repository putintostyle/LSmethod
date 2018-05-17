
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
for i in range(180,181):
    File.append('IM'+str(i))
                                        
def image_feature(image, image_size):
    center = 256
    location_x = np.absolute(np.array([np.arange(image_size).tolist() for i in range(image_size)])-center)/center *(image != 0)
    location_y = np.absolute(np.array([(i*np.ones(image_size)).tolist() for i in range(image_size)])-center)/center *(image != 0)
    image = preprocessing.scale(np.array(image)) *(image != 0)
    image_x = np.gradient(image)[1] *(image != 0)
    image_y = np.gradient(image)[0] *(image != 0)
    #image_xx = np.gradient(image_x)[1]
    #image_yy = np.gradient(image_y)[0]
    #image_xy = np.gradient(image_x)[0]
    #curvature_image = preprocessing.scale((image_x**2*image_yy + image_xx*image_y**2 - 2*image_xy)/((image_x**2+image_y**2 + 1e-04)**(3/2)))
    #det = image_xx * image_yy - image_xy**2
    r = np.sqrt((location_x)**2+(location_y)**2) / (np.max(np.sqrt((location_x)**2+(location_y)**2)))
    theta = np.arctan2(location_y,location_x) / (np.max(np.arctan2(location_y,location_x))-np.min(np.arctan2(location_y,location_x)))
    
    return[location_x,location_y,image,r,np.absolute(theta),image_x,image_y]#,curvature_image]
def local_feature(image):
    feature = []
    boundary_x = np.zeros((1,len(image)+2))
    boundary_y = np.zeros((len(image),1))
    image = np.append(boundary_y,image,axis = 1)
    
    image = np.append(image,boundary_y,axis = 1)
    image = np.append(boundary_x,image,axis = 0)
    image = np.append(image,boundary_x,axis = 0)
    
    for i in range(1,len(image)-1):
        for j in range(1,len(image)-1):
            feature.append([image[i-1:i+2,j-1:j+2].flatten()])
    return np.array(feature)
def Gaussian_Mixture_Model(feature_matrix,cluster): # x,y,I,Ix,Iy,Ixy,Ixx,Iyy,k,det,r,theta
    #feature_5 = local_feature(feature_2)
    
    #feature_extra = local_feature(feature_matrix[2])
    #print(np.shape(feature_5[0]))
    feature_0 = feature_matrix[0].flatten() #x
    feature_1 = feature_matrix[1].flatten() #y
    feature_2 = feature_matrix[2].flatten() #I
    feature_3 = feature_matrix[3].flatten() #r
    feature_4 = feature_matrix[4].flatten() #theta
    
    #feature_5 = feature_matrix[5].flatten()  #I_x
    #feature_6 = feature_matrix[6].flatten()  #I_y
    # feature_7 = feature_matrix[7].flatten()
    # feature_8 = feature_matrix[8].flatten()
    # feature_9 = feature_matrix[9].flatten()
    # feature_10 = feature_matrix[10].flatten()
    # feature_11 = feature_matrix[11].flatten()
    X = [np.array([feature_2[i],feature_3[i]]) for i in range(0,len(feature_2))]

    #mean_initial = initail_mean(feature_matrix,cluster)
    gmm_mean = np.transpose(GMM(n_components=cluster).fit(X).means_)
    
    #### fmin ####
    Y = solve_matrix_lsq(X,gmm_mean)
    #Y = np.matmul(np.linalg.pinv(gmm_mean),np.transpose(X))
    #labels = gmm.predict(X)
    return [gmm_mean,Y]

def initail_mean(image,cluster):
    mean = []
    for j in range(cluster):
        tmp0 = []
        tmp1 = []
        tmp2 = []
        tmp3 = []
        tmp4 = []
        tmp5 = []
        tmp6 = []
        for i in range(len(image[2][128])):        
            if (image[2][128][i] > j/cluster * max(image[2][128])) & (image[2][128][i] <= (j+1)/cluster*max(image[2][128])):
                # tmp0.append(image[0][128][i])
                # tmp1.append(image[1][128][i])
                tmp2.append(image[2][128][i])
                tmp3.append(image[3][128][i])
                tmp4.append(image[4][128][i])
                # tmp5.append(image[5][128][i] + 1E-10)
                # tmp6.append(image[6][128][i] + 1E-10)
                #extra.append(image[2][128][i],image[2][128][i],image[2][128][i],image[2][128][i],image[2][128][i],image[2][128][i],image[2][128][i],image[2][128][i],image[2][128][i])
        #erxtra = [np.mean(tmp2) for k in range(9)]
        #print(erxtra)
        tmp = np.array([np.mean(tmp2),np.mean(tmp3),np.mean(tmp4)])
        mean.append(tmp)  
    return np.array(mean)      
     

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
    label = np.zeros((512,512))
    for chart_no in range(1):
        Y = Gaussian_Mixture_Model(image_feature_data,CLUSTER)[1]
        # location = Gaussian_Mixture_Model(atalas_class[chart_no],CLUSTER)[0]
        # print(location)
        # print(Y[0])
        label_local = []
        
        for i in range(len(Y[0])):
            #print(Y[:,i])
            #print(np.where(np.amax(Y[:,i]))[0][0])
            #np.append(label,(np.where(np.amax(Y[:,i]))[0][0]))
            tmp = (Y[:,i]).tolist() 
            label_local.append(tmp.index(max(tmp)))#+chart_no*CLUSTER)
        #print(label)
    label_local = np.reshape(label_local,(512,512))
        # if chart_no == 0:
        #     label[0:256,0:256] = np.array(label_local)
        # elif chart_no == 1:
        #     label[0:256,256:512] = np.array(label_local)
        # elif chart_no == 2:
        #     label[256:512,0:256] = np.array(label_local)
        # elif chart_no == 3:
        #     label[256:512,256:512] = np.array(label_local)   
    
    for i in range(CLUSTER*0,CLUSTER*1):
        image = ds_pixel*(1-((label_local==i).astype(np.int)))
        #plt.imshow(image,cmap = plt.cm.bone)
        plt.suptitle("The label" + str(i))
        
        plt.imsave("Cluster" + str(i)+".png",image,cmap = plt.cm.bone,  dpi = 1500)
        #plt.show() 
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