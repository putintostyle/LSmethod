
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
                                        

def main(file):
    ds=dicom.read_file('D:\\MRI_segmentation\\T1 3D\\SE12\\'+file)#+'.dcm')
    #ds=dicom.read_file(file)#+'.dcm')
    
    
    pixel_size=len(ds.pixel_array)
     
    ds_pixel = ds.pixel_array
    #phi = (initial_level_set(pixel_size,390,130,20)) #390 130 20
    for i in range(pixel_size):
        for j in range(pixel_size):
            if ds_pixel[i][j]>300:
                ds_pixel[i][j] = 0
    plt.imshow(ds_pixel, cmap = plt.cm.bone)
    plt.show()

start=time.time()

for i in File:
    main(i)
end=time.time()
print(end-start)