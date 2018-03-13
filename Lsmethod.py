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
                                        

def fdm(image, image_size, phi, f, lambda_1, lambda_2, windowsize):
    eps = 1e-04
    for i in range(1,image_size-1):
        for j in range (1,image_size-1):
            fox = (phi[i][j]-phi[i-1][j]) #fx
            foy = (phi[i][j]-phi[i][j-1]) #fy
            sox = (-2*phi[i][j]+phi[i-1][j]+phi[i][j-1]) #fxx
            soy = (-2*phi[i][j]+phi[i][j-1]+phi[i-1][j]) #fyy
            soxy = 1/2*(phi[i+1][j+1]-phi[i+1][j-1]-phi[i-1][j+1]+phi[i-1][j-1]) #fxy
            length_23 = (fox**2+foy**2)**(2/3) #(fx^2 +fy^2)
            indicate = eps/(3.14*eps**2 + phi[i][j]**2) #delta function
            loce = 0 #local energy
            local_image = []
            for k in range(0-(windowsize-1)/2,1+(windowsize-1)/2) :
                for l in range(0-(windowsize-1)/2,1+(windowsize-1)/2): 
                    if((k-i>0)&&((l-j)>0)):
                        local_image.append(image[k-i][l-j])
                    else:
                        local_image.append("Null")

            f = local_f(local_image,windowsize)

            for k in range(0-(windowsize-1)/2,1+(windowsize-1)/2) :
                for l in range(0-(windowsize-1)/2,1+(windowsize-1)/2): 
                    loce+=constrain*((1+f[k][l]+f[k][l]**2 - image[i][j])**2) * ((k)**2+(l)**2)/4       

            phi[i][j] -= indicate*((lambda_2*(fox*fox*sox+foy*foy*soy-fox*foy*soxy)/length_23)+loce+lambda_1)
    return phi

def local_f(local_image, windowsize):
    #index = [[0]*6 for i in range(windowsize)]
    A = []
    index = []
    for x in range(0-int((windowsize-1)/2),1+int((windowsize-1)/2)):
        for y in range(0-int((windowsize-1)/2),1+int((windowsize-1)/2)):
            index.append([x,y])
    #print(index)
    for element in index:
        x = element[0]
        y = element[1]
        tmp = [1,x,y,x*y,x**2,y**2]
        A.append(tmp) 
    #A = [[1,1,0,1,0,0],[1,0,1,0,0,1],[1,-1,0,-1,0,0],[1,0,-1,0,0,-1],[1,1,1,1,1,1],[1,1,-1,1,-1,1],[1,-1,1,1,-1,1],[1,-1,-1,1,1,1]]
    
    b = np.reshape(local_image,(windowsize**2,1))
    for pixel in b:
        if b[pixel] = "Null"
            A[pixel] = [0 for i in range(windowsize**2)]
            b[pixel] = 0
    At = np.transpose(A)

    AtA = np.matmul(At,A)

    Atb = np.matmul(At,b)

    coef = np.linalg.solve(AtA,b)
    f = []
    for element in index:
        x = element[0]
        y = element[1]
        tmp = [1*coef[0],x*coef[1],y*coef[2],x*y*coef[3],x**2*coef[4],y**2*coef[5]]
        f.append(tmp)
    return np.reshape(f,(windowsize,windowsize))
def main(file):
    ds=dicom.read_file('D:\\MRI_segmentation\\T1 3D\\SE12\\'+file)#+'.dcm')
    #ds=dicom.read_file(file)#+'.dcm')
    
    
    pixel_size=len(ds.pixel_array)
    
    #ds_pixel=threads_hold(ds.pixel_array,compute_sta(ds.pixel_array)[0]+0.1*compute_sta(ds.pixel_array)[1],"denoise")
    
    ds_pixel = ds.pixel_array

