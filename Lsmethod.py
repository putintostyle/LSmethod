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
                                        

def fdm(image, image_size, phi, lambda_1, lambda_2, windowsize):
    eps = 1e-02

    for i in range(1,image_size):
        for j in range (1,image_size):
            if image[i][j]!=0:
                fox = (phi[i][j]-phi[i-1][j]) #fx
                foy = (phi[i][j]-phi[i][j-1]) #fy
                if (fox!=0)&(foy!=0):
                    sox = (-2*phi[i][j]+phi[i-1][j]+phi[i+1][j]) #fxx
                    soy = (-2*phi[i][j]+phi[i][j-1]+phi[i][j+1]) #fyy
                    soxy = 1/2*(phi[i+1][j+1]-phi[i+1][j-1]-phi[i-1][j+1]+phi[i-1][j-1]) #fxy
                    length = (fox*fox+foy*foy) #(fx^2 +fy^2)
                    length_23 = length**(1.5)
                    #delta function
                    #if indicate >= 5:
                    #    indicate = 1
    
                indicate = eps/(3.14*(eps*eps + phi[i][j]*phi[i][j]))
                loce = 0 #local energy
                local_image = []
                local_index = []
                #mean_container = []
                for k in range(0-int((windowsize-1)/2),1+int((windowsize-1)/2)) :  #local energy setting
                    for l in range(0-int((windowsize-1)/2),1+int((windowsize-1)/2)):
                        if (phi[i-k][j-l]>=0)&((i-k)>0)&((j-l)>0)&((i-k)<image_size)&((j-l)<image_size):
                            local_index.append([k,l,1])
                            if(((i-k)>0)&((j-l)>0)&((i-k)<image_size)&((j-l)<image_size)):
                                local_image.append(image[i-k][j-l])
                                #mean_container.append(image[i-k][j-l])
                            else:
                                local_image.append("Null")
                                
                        else:
                            local_index.append([k,l,0])
                            local_image.append("Null")

                        
                
                start_local_time = time.time()
                f = local_f(local_image,windowsize,local_index)  #compute local energy
                
                end_local_time = time.time()
                #print(f)
                time_loc = start_local_time-end_local_time
                """print("location_x = "+str(i)+"\n")
                print("location_y = "+str(j)+"\n")
                print("loacl_energy_computation_time = " + str(abs(time_loc)) + "\n")
                """
    
                for k in range(0-int((windowsize-1)/2),1+int((windowsize-1)/2)) :
                    for l in range(0-int((windowsize-1)/2),1+int((windowsize-1)/2)): 
                        loce += ((1+f[k+int((windowsize-1)/2)][l+int((windowsize-1)/2)]+0.5*f[k+int((windowsize-1)/2)][l+int((windowsize-1)/2)] - image[i][j]))**2 * ((k)**2+(l)**2)/(windowsize*windowsize)
                
                   
                #loce = (np.mean(mean_container) - image[i][j])**2
                if (fox!=0)&(foy!=0):
                    phi[i][j] += indicate*((lambda_2*(fox*fox*sox+foy*foy*soy-fox*foy*soxy)/length_23)-loce-lambda_1)
                else:
                    phi[i][j] -= indicate*(loce+lambda_1)
    return phi

def local_f(local_image, windowsize,index):
    
    #index = [[0]*6 for i in range(windowsize)]
    A = []
    for element in index:
        x = element[0]
        y = element[1]
        tmp = [1,x,y,x*y,x**2,y**2]
        A.append(tmp) 
    #A = [[1,1,0,1,0,0],[1,0,1,0,0,1],[1,-1,0,-1,0,0],[1,0,-1,0,0,-1],[1,1,1,1,1,1],[1,1,-1,1,-1,1],[1,-1,1,1,-1,1],[1,-1,-1,1,1,1]]
    
    b = local_image
    
    index_b = []
    for pixel in range(0,len(b)):
        if b[pixel] == "Null" :
            index_b.append(pixel)
    
    A = np.delete(A,index_b,axis = 0)
    b = np.delete(b,index_b,None)
    b = b.astype(float)
    At = np.transpose(A)

    AtA = np.matmul(At,A)
   
    Atb = np.matmul(At,b)
    
    if np.linalg.det(AtA) == 0:
        #coef = np.zeros((6),dtype = int)
        f = np.zeros((windowsize**2), dtype = int)
    else:
        coef = np.linalg.solve(AtA,Atb)
        f = []
        
        for element in index:
            if element[2] == 1:
                x = element[0]
                y = element[1]
                tmp = 1*coef[0]+x*coef[1]+y*coef[2]+x*y*coef[3]+x**2*coef[4]+y**2*coef[5]
                f.append(tmp)
            else:
                f.append(0)
    return np.reshape(f,(windowsize,windowsize))


def initial_level_set(image_size, initial_value_1, initial_value_2):
    phi_graph = [[0]*image_size for i in range(image_size)]
    
    for i in range(0,image_size):
        for j in range(0,image_size):
            data = -((350-i)**2+(300-j)**2)+40**2
            if abs(data) < 7**2:
                phi_graph[i][j] = 0
            else:
                phi_graph[i][j] = data
            """if (i>127+50)&(i<384+50)&(j>127)&(j<384):
                phi_graph[i][j] = initial_value_1
            elif(((i==127+50)&((j>128)&(j<384))) | ((i==256+50)&((j>128)&(j<384))) | ((j==256)&((i>128+50)&(i<384+50))) | ((j==127)&((i>128+50)&(i<384+50)))):
                phi_graph[i][j] = 0
            else:
                phi_graph[i][j] = initial_value_2"""
    return phi_graph

def main(file):
    ds=dicom.read_file('D:\\MRI_segmentation\\T1 3D\\SE12\\'+file)#+'.dcm')
    #ds=dicom.read_file(file)#+'.dcm')
    
    
    pixel_size=len(ds.pixel_array)
    
    #ds_pixel=threads_hold(ds.pixel_array,compute_sta(ds.pixel_array)[0]+0.1*compute_sta(ds.pixel_array)[1],"denoise")
    
    ds_pixel = np.array(ds.pixel_array).astype(int)
    
    phi = initial_level_set(pixel_size, 10, -10)
    for i in range(5):
        
        phi = fdm(ds_pixel,pixel_size,phi, 1,1,5)
    
    image = [[0]*pixel_size for i in range(pixel_size)]
    for i in range(pixel_size):
        for j in range(pixel_size):
            if phi[i][j]>0:
                image[i][j] = ds_pixel[i][j]
    
    plt.imshow(image, cmap = plt.cm.bone)
    plt.show()


start=time.time()

for i in File:
    main(i)
end=time.time()
print(end-start)