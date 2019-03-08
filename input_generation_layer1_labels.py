'''
Original Dmatrix 1: right, 2: down, 3: left, 4: up
Sung Eun Kim, Jonghyun Harry Lee
'''

import sys
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import h5py

rnet = loadmat('input_0.0001.mat', squeeze_me=True)

# size to 12 by 12 
img=[]
size=(12,12)
z_padding=1

for i in range(1000):
    data = rnet['D%04d'% (i+1)]
    data2 = np.zeros(size)
    data2[z_padding:size[0],z_padding:size[1]] = data
    img=[*img,data2]

# prepare 5 channels
img_orig=np.array(img)
img0=np.array(img)
img1=np.copy(img0)
img2=np.copy(img0)
img3=np.copy(img0)
img4=np.copy(img0)
# img0 left as is
# img1 for 1
img1[img_orig==1] = 1  
img1[img_orig==2] = 0 
img1[img_orig==3] = 0 
img1[img_orig==4] = 0 
# img2 for 2
img2[img_orig==1] = 0 
img2[img_orig==2] = 1 
img2[img_orig==3] = 0 
img2[img_orig==4] = 0 
# img3 for 3
img3[img_orig==1] = 0 
img3[img_orig==2] = 0 
img3[img_orig==3] = 1 
img3[img_orig==4] = 0 
# img3 for 3
img4[img_orig==1] = 0 
img4[img_orig==2] = 0 
img4[img_orig==3] = 0 
img4[img_orig==4] = 1 

ti = np.zeros((1000,12,12,5))
ti[:,:,:,0] = img0
ti[:,:,:,1] = img1
ti[:,:,:,2] = img2
ti[:,:,:,3] = img3
ti[:,:,:,4] = img4

savemat('input_layer1_labels.mat', dict([('ti', ti)]))

