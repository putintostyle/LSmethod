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
                                        

def fdm(image, image_size, phi, lambda_1, lambda_2, windowsize,its):
    image = np.int_(image)
    eps = 1
    
    for i in range(0,image_size):
        for j in range (0,image_size):
            if image[i][j]!=0:
                fox = 1/2*(phi[i][j+1]-phi[i][j-1]) #fx
                foy = 1/2*(phi[i+1][j]-phi[i-1][j]) #fy
                if (fox!=0)|(foy!=0):
                    sox = (-2*phi[i][j]+phi[i][j-1]+phi[i][j+1]) #fxx
                    soy = (-2*phi[i][j]+phi[i-1][j]+phi[i+1][j]) #fyy
                    soxy = 1/4*(phi[i+1][j+1]-phi[i+1][j-1]-phi[i-1][j+1]+phi[i-1][j-1]) #fxy
                    length = (fox*fox+foy*foy) #(fx^2 +fy^2)
                    length_23 = length**1.5
                    #delta function
                    #if indicate >= 5:
                    #    indicate = 1
                    
                    #print("local kappa in (" + str(i) + ',' + str(j) + ") is " + str((fox*fox*soy+foy*foy*sox-2*fox*foy*soxy)/length_23) + "\n" )
                indicate = (eps/(3.14*(eps*eps + phi[i][j]*phi[i][j])))
                
                loce = 0.0 #local energy
                local_image = []
                local_index = []
                mean_container = []
                for k in range(0-int((windowsize-1)/2),1+int((windowsize-1)/2)) :  #local energy setting
                    for l in range(0-int((windowsize-1)/2),1+int((windowsize-1)/2)):
                        if (phi[i-k][j-l]>=0)&((i-k)>0)&((j-l)>0)&((i-k)<image_size)&((j-l)<image_size):
                            local_index.append([k,l,1])
                            if(((i-k)>0)&((j-l)>0)&((i-k)<image_size)&((j-l)<image_size)):
                                local_image.append(image[i-k][j-l])
                                mean_container.append(image[i-k][j-l])
                            else:
                                local_image.append("Null")
                                mean_container.append(0)

                                
                        else:
                            local_index.append([k,l,0])
                            local_image.append("Null")
                            mean_container.append(0)

                        
                
                start_local_time = time.time()
                #f = local_f(local_image,windowsize,local_index)  #compute local energy
                
                #end_local_time = time.time()
                #print(f)
                #time_loc = start_local_time-end_local_time
                
    
                #for k in range(0-int((windowsize-1)/2),1+int((windowsize-1)/2)) :
                    #for l in range(0-int((windowsize-1)/2),1+int((windowsize-1)/2)):
                        #difference = (f[k+int((windowsize-1)/2)][l+int((windowsize-1)/2)]-image[i][j])
                        #print("difference = " + str(difference)) 
                        #loce += (1+difference + difference**2 /2)#* ((k)**2+(l)**2))/(windowsize*windowsize)
                loce = abs((np.mean(mean_container) - image[i][j]))
                if (i > 150)&(i<160)&(j>150)&(j<160):
                    print("local energy in (" + str(i) + ',' + str(j) + ") is " + str(loce) + "\n" )   
                

                if((fox!=0)|(foy!=0)):
                    #if its>8:
                    if abs(phi[i][j])<5:
                        print("origin phi in (" + str(i) + ',' + str(j) + ") is " +str(phi[i][j]) + "\n" ) #debug
                    phi[i][j] += indicate*((lambda_2*(fox*fox*soy+foy*foy*sox-2*fox*foy*soxy)/length_23)-loce-lambda_1)
                    #if its>8:
                    if abs(phi[i][j])<5:
                        print("modify phi in (" + str(i) + ',' + str(j) + ") is " +str(phi[i][j]) + "\n" )
                    #print("kappa in (" + str(i) + ',' + str(j) + ") is " +str((fox*fox*sox+foy*foy*soy-fox*foy*soxy)/length_23) + "\n" )
                else:
                    #if its>8:
                    if abs(phi[i][j])<5:
                        print("origin phi in (" + str(i) + ',' + str(j) + ") is " +str(phi[i][j]) + "\n" )
                    phi[i][j] += indicate*(-loce-lambda_1)
                    #if its>8:
                    if abs(phi[i][j])<5:
                        print("modify phi in (" + str(i) + ',' + str(j) + ") is " +str(phi[i][j]) + "\n" )
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


def initial_level_set(image_size,radius,center_x, center_y):
    phi_graph = [[0]*image_size for i in range(image_size)]

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
    a = np.float64(255)
    maxi = np.matrix(image).max()
    mini = np.matrix(image).min()
    image = a*np.array(image)/(maxi-mini)
    
    return image
            
def main(file):
    ds=dicom.read_file('D:\\MRI_segmentation\\T1 3D\\SE12\\'+file)#+'.dcm')
    #ds=dicom.read_file(file)#+'.dcm')
    
    
    pixel_size=len(ds.pixel_array)
    
    #ds_pixel=threads_hold(ds.pixel_array,compute_sta(ds.pixel_array)[0]+0.1*compute_sta(ds.pixel_array)[1],"denoise")
    
    ds_pixel = re_scaling(ds.pixel_array)
    
    phi = (initial_level_set(pixel_size,50,150,140))

    
    for i in range(1):
        
        phi = fdm(ds_pixel,pixel_size,phi, 1,1,3,i)
    
    image = [[0]*pixel_size for i in range(pixel_size)]
    for i in range(pixel_size):
        for j in range(pixel_size):
            if phi[i][j] > 0 :
                image[i][j] = ds_pixel[i][j]
    print(phi>0)
    plt.imshow(np.float64(image), cmap = plt.cm.bone)
    plt.show()




start=time.time()

for i in File:
    main(i)
end=time.time()
print(end-start)