
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
#File=['IM1.dcm']
File=[]
start=time.time()
for i in range(120,121):
    File.append('IM'+str(i))
                                        

def image_feature(image, image_size):
    center = 256
    location_x = np.array([np.arange(image_size).tolist() for i in range(image_size)])
    location_y = np.array([(i*np.ones(image_size)).tolist() for i in range(image_size)])
    image = np.array(image)
    image_x = np.gradient(image)[1]
    image_y = np.gradient(image)[0]
    image_xx = np.gradient(image_x)[1]
    image_yy = np.gradient(image_y)[0]
    image_xy = np.gradient(image_x)[0]
    curvature_image = (image_x**2*image_yy + image_xx*image_y**2 - 2*image_xy)/((image_x**2+image_y**2 + 1e-04)**(3/2))
    det = image_xx * image_yy - image_xy**2
    r = np.sqrt((location_x-center)**2+(location_y-center)**2)
    theta = np.arctan2(location_y-center,location_x-center)* 180 / np.pi
    return[location_x,location_y,image,image_x,image_y, image_xx, image_yy, image_xy, curvature_image,det,r,theta]
    #return[location_x/512,location_y/512,image/(np.amax(image) - np.amin(image))]#,r/(np.amax(r) - np.amin(r))]
def data_visualization(feature_matrix): # x,y,I,Ix,Iy,Ixy,Ixx,Iyy,k,det,r,theta
    feature_0 = feature_matrix[0].flatten() #x
    feature_1 = feature_matrix[1].flatten() #y
    feature_2 = feature_matrix[2].flatten() #I
    feature_3 = feature_matrix[3].flatten() #Ix
    feature_4 = feature_matrix[4].flatten() #Iy
    feature_5 = feature_matrix[5].flatten() #Ixy
    feature_6 = feature_matrix[6].flatten() #Ixx
    feature_7 = feature_matrix[7].flatten() #Iyy
    feature_8 = feature_matrix[8].flatten() #k
    feature_9 = feature_matrix[9].flatten() #det
    feature_10 = feature_matrix[10].flatten() #r
    feature_11 = feature_matrix[11].flatten() #theta
    plt.scatter(feature_10[abs(feature_11+135)<5],feature_2[abs(feature_11+135)<5]/(np.amax(feature_2)-np.amin(feature_2)))
    #plt.hist(feature_2[feature_2>100])
    plt.show()
    #print(feature_11)
    #X = [[feature_0[i],feature_1[i],feature_2[i]] for i in range(0,len(feature_0))]#,feature_3[i],feature_4[i],feature_5[i],feature_6[i],feature_7[i],feature_8[i],feature_9[i],feature_10[i],feature_11[i]] for i in range(0,len(feature_0))]
    #gmm = GMM(n_components=10).fit(X)
    #labels = gmm.predict(X)
    #return labels

def main(file):
    ds=dicom.read_file('D:\\MRI_segmentation\\T1 3D\\SE12\\'+file)#+'.dcm')
    #ds=dicom.read_file(file)#+'.dcm')
    
    
    pixel_size=len(ds.pixel_array)
     
    ds_pixel = ds.pixel_array
    #print(ds_pixel)
    #phi = (initial_level_set(pixel_size,390,130,20)) #390 130 20

    image_feature_data = image_feature(ds_pixel,pixel_size)
    data_visualization(image_feature_data)
    #print(label[512*200+25])
    #segment = np.array(ds_pixel)*((1-(np.reshape((label == 8).astype(np.int),(512,512)))) + (1-(np.reshape((label == 9).astype(np.int),(512,512)))))
    #for i in range(10):
    #    segment = ((np.reshape((label == i).astype(np.int),(512,512))))
    #    plt.imshow(segment,cmap = plt.cm.bone)
    #    plt.suptitle("this is "+str(i))
    #    plt.show()  
    #print(ds_pixel*(image_feature_data[11]==45))  
    plt.imshow(ds_pixel*(abs(image_feature_data[11]+135)<5),cmap = plt.cm.bone)
    plt.show()




start=time.time()

for i in File:
    main(i)
end=time.time()
print(end-start)