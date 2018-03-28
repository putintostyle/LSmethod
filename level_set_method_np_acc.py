# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 22:49:15 2018

@author: User
"""

import dicom
import matplotlib.pyplot as plt
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
                                        
def convert2polar(x,y,image_size,center):
    r = np.sqrt((x-center)**2+(y-center)**2)
    theta = np.arctan2(y-center,x-center)* 180 / np.pi
    return [r,theta]

def image_feature(image, image_size):
    location_x = np.array([np.arange(image_size).tolist() for i in range(image_size)])
    location_y = np.array([(i*np.ones(image_size)).tolist() for i in range(image_size)])
    image = np.array(image)
    image_x = np.gradient(image)[1]
    image_y = np.gradient(image)[0]
    image_xx = np.gradient(image_x)[1]
    image_yy = np.gradient(image_y)[0]
    image_xy = np.gradient(image_x)[0]
    curvature_image = (image_x**2*image_yy + image_xx*image_y**2 - 2*image_xy)/((image_x**2+image_y**2 + 1e-04)**(3/2))
    return[location_x,location_y,image_x,image_y, image_xx, image_yy, image_xy, curvature_image]

def fdm(image, image_feature,polar_coordinate, image_size, phi, lambda1, lambda2, lambda3, lambda4):
    #image = np.int_(image)
    #x 方向是j的位置
    #y 方向是i的位置
    eps = 1
    phi = np.array(phi) 
    phi_x = np.gradient(phi)[1] #他跟我們的直覺不一樣 1 是x方向
    phi_y = np.gradient(phi)[0]#.transpose()
    phi_xx = np.gradient(phi_x)[1]
    phi_yy = np.gradient(phi_y)[0]
    phi_xy = np.gradient(phi_x)[0]
    #phi_yx = np.gradient(phi_y)[1]
    curvature_phi = (phi_x**2*phi_yy + phi_xx*phi_y**2 - 2*phi_xy)/((phi_x**2+phi_y**2 + 1e-04)**(3/2))
    #regularization = curvature*lambda1
    #for i in range(0,511):
        #for j in range (0,511):
    delta_function = (eps/(3.14*(eps*eps + phi**2)))

    ######################            DATA FEATURE            ######################
    
    location_x = image_feature[0]
    location_y = image_feature[1]
    image = np.array(image)
    image_x = image_feature[2]
    image_y = image_feature[3]
    image_xx = image_feature[4]
    image_yy = image_feature[5]
    image_xy = image_feature[6]
    curvature_image = image_feature[7]
    
    ######################             Polar coordinate        #######################
    r = polar_coordinate[0]
    theta = polar_coordinate[1]


    
    Hea = 0.5*(1 + (2 / 3.14)*np.arctan(phi/eps))


    s1=Hea*image 
    s2=(1-Hea)*image 
    s3=1-Hea 
    C1 = s1.sum()/ Hea.sum() 
    C2 = s2.sum()/ s3.sum() 

    phi = phi + delta_function*(lambda1*curvature_phi - lambda2 - lambda3*(image - C1)**2 + lambda4*(image - C2)**2)
      
    return phi



def initial_level_set(image_size,center_x, center_y,radius):
    #phi_graph = np.ones((image_size,image_size),np.int8)
    phi_graph = np.zeros((image_size,image_size))
    """a = (center_x-int(radius/2))
    b = (center_x+int(radius/2))
    c = (center_y-int(radius/2))
    d = (center_y+int(radius/2))
    phi_graph[a:b,c:d] = -1
    phi_graph = -phi_graph"""


    for i in range(0,image_size):
        for j in range(0,image_size):
            data = math.sqrt((center_y-i)**2+(center_x-j)**2)-radius
            
            if abs(int(data)) == 0:
                phi_graph[i][j] = 0
            elif int(data) > 0:
                phi_graph[i][j] = -data
            elif int(data) < 0:
                phi_graph[i][j] = -data
    return phi_graph

def re_scaling(image):
    #print(image[200][200].dtype)
    #print("GOOOOOOOOOO")
    a = (255)
    maxi = np.matrix(image).max()
    
    mini = np.matrix(image).min()
    
    image = np.uint8((np.array(image)/(maxi-mini)*255))
    
    return image
            
def main(file):
    ds=dicom.read_file('D:\\MRI_segmentation\\T1 3D\\SE12\\'+file)#+'.dcm')
    #ds=dicom.read_file(file)#+'.dcm')
    
    
    pixel_size=len(ds.pixel_array)
    #plt.imshow((ds.pixel_array))
    #plt.show()
    #print(ds.pixel_array.dtype)
    #ds_pixel=threads_hold(ds.pixel_array,compute_sta(ds.pixel_array)[0]+0.1*compute_sta(ds.pixel_array)[1],"denoise")
    
    ds_pixel = ds.pixel_array#re_scaling(ds.pixel_array)
    
    phi = (initial_level_set(pixel_size,250,250,150)) #390 130 20

    image_feature_data = image_feature(ds_pixel,pixel_size)
    polar_coordinate = convert2polar(image_feature_data[0],image_feature_data[1],pixel_size,int(pixel_size/2))
    for it in range(5):
        
        phi = fdm(ds_pixel,image_feature_data,polar_coordinate,pixel_size,phi, 1,1,1.5,0.02)
    
        image = [[0]*pixel_size for i in range(pixel_size)]
        for i in range(pixel_size):
            for j in range(pixel_size):
                if phi[i][j] > 0 :
                    image[i][j] = ds_pixel[i][j]
        image = np.array(image)
        
        #print(ds_pixel.dtype)
        plt.imshow(image,cmap = plt.cm.bone)
        plt.suptitle("this is "+str(it))
        plt.show()




start=time.time()

for i in File:
    main(i)
end=time.time()
print(end-start)