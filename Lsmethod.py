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
import cv2
import copy
#File=['IM1.dcm']
File=[]
start=time.time()
for i in range(120,121):
    File.append('IM'+str(i))
                                        

def fdm(image, size, phi, f, lambda_1, lambda_2):
    eps = 1e-04
    for i in range(0,size):
        for j in range (0,size):
            fox = (phi[i][j]-phi[i-1][j])
            foy = (phi[i][j]-phi[i][j-1])
            sox = (-2*phi[i][j]+phi[i-1][j]+phi[i][j-1])
            soy = (-2*phi[i][j]+phi[i][j-1]+phi[i-1][j])
            soxy = 1/2*(phi[i+1][j+1]-phi[i+1][j-1]-phi[i-1][j+1]+phi[i-1][j-1])
            length_23 = (fox**2+foy**2)**(2/3)
            indicate = eps/(3.14*eps**2 + phi[i][j]**2)
            loce = 0
            for k in range(-1,2,2):
                for l in range(-1,2,2):
                    loce+=((1+f[k-i][l-j]+f[k-i][l-j]**2 - image[i][j])**2) * ((k-i)**2+(l-j)**2)/4
            phi[i][j] -= indicate*((lambda_2*(fox*fox*sox+foy*foy*soy-fox*foy*soxy)/length_23)+loce+lambda_1)
            


def main(file):
    ds=dicom.read_file('D:\\MRI_segmentation\\T1 3D\\SE12\\'+file)#+'.dcm')
    #ds=dicom.read_file(file)#+'.dcm')
    
    
    pixel_size=len(ds.pixel_array)
    
    #ds_pixel=threads_hold(ds.pixel_array,compute_sta(ds.pixel_array)[0]+0.1*compute_sta(ds.pixel_array)[1],"denoise")
    
    ds_pixel = ds.pixel_array