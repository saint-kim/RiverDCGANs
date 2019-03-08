'''
Original Dmatrix 1: right, 2: down, 3: left
Sung Eun Kim, Jonghyun Harry Lee
'''

import sys
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import h5py

beta = 1000 # 0.0001
rnet = loadmat(('input_%d.mat' % beta), squeeze_me=True)

# size to 12 by 12 
img=[]
size=(12,12)
z_padding=1

for i in range(1000):
    data = rnet['D%04d'% (i+1)]
    if (data == 4).any(): # for now..
        pass
    else:
        data2 = np.zeros(size)
        data2[z_padding:size[0],z_padding:size[1]] = data
        img=[*img,data2]

# prepare 3 channels
img_orig=np.array(img)
img0=np.array(img)
img1=np.copy(img0)
img2=np.copy(img0)

# img0 for right
img0[img_orig==1] = 1  
img0[img_orig==2] = 1 
img0[img_orig==3] = 0 

# img1 for left
img1[img_orig==1] = 0  
img1[img_orig==2] = 1 
img1[img_orig==3] = 1 

ti = np.zeros((img0.shape[0],img0.shape[1],img0.shape[2],2))
ti[:,:,:,0] = img0
ti[:,:,:,1] = img1

savemat(('input_lr_layer2_%d.mat' % beta), dict([('ti', ti)]))

