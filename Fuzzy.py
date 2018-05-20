
# -*- coding: utf-8 -*-
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

def image_feature(image, image_size):
    center = 256
    #location_x = (np.array([np.arange(image_size).tolist() for i in range(image_size)])-center)/center
    #location_y = (np.array([(i*np.ones(image_size)).tolist() for i in range(image_size)])-center)/center
    image = preprocessing.scale(np.array(image))
    image_x = np.gradient(image)[1]
    image_y = np.gradient(image)[0]
    image_xx = np.gradient(image_x)[1]
    image_yy = np.gradient(image_y)[0]
    image_xy = np.gradient(image_x)[0]
    curvature_image = preprocessing.scale((image_x**2*image_yy + image_xx*image_y**2 - 2*image_xy)/((image_x**2+image_y**2 + 1e-04)**(3/2)))
    det = image_xx * image_yy - image_xy**2
    return[image]#,curvature_image]
    
def main(file):
    ds=dicom.read_file('D:\\MRI_segmentation\\T1 3D\\SE12\\'+file)#+'.dcm')
    pixel_size=len(ds.pixel_array)
     
    ds_pixel = ds.pixel_array

    image_feature_data = image_feature(ds_pixel,pixel_size)
    fft_image = (np.fft.fft2(ds_pixel))
    print(np.amax(fft_image))
    print(np.amin(fft_image))
    FILTER = fft_image*((np.absolute((np.log(np.absolute(fft_image))-12))>0.05).astype(np.int))
    plt.imshow(np.fft.ifft2(FILTER).astype(np.int), cmap = 'gray')
    plt.show()
    plt.imsave("tmp.png",ds_pixel,cmap = plt.cm.bone, dpi = 1000)
    #####################
    # NUMBER OF CLUSTER #
    ##################### 
    CLUSTER = 3



start=time.time()

for i in File:
    main(i)
end=time.time()
print(end-start)