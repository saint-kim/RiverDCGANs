#from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam


import matplotlib.pyplot as plt
import sys
from scipy.io import loadmat, savemat
import numpy as np
import h5py
from keras.models import load_model

def ths(data,nlevel,minv=-1.0,maxv=1.0):
    '''
    thresholding for nlevels
    '''
    #mindata = data.min()
    #maxdata = data.max()
    data_tmp = np.copy(data)
    
    thsvals = np.linspace(minv,maxv,nlevel+1)
    minval = -1000.
    for i, thsval in enumerate(thsvals):
        data_tmp[(data >= minval) & (data < thsval)] = i-1
        minval = thsval
    
    return data_tmp


def convert(img):
    img_final = np.zeros((12,12),dtype=int)
    img1 = np.copy(img)
    img01= img1[:,:,0]*img1[:,:,1] # left & right => could be up or down
    
    img_final[(img1[:,:,0]==1) & (img01 == 0)] = 1 # left
    img_final[(img1[:,:,1]==1) & (img01 == 0)] = 3 # right
    img_final[(img1[:,:,2]==1) & (img01 == 1)] = 4 # down
    img_final[(img1[:,:,2]==0) & (img01 == 1)] = 2 # up
    
    return img_final

generator=load_model('./Case_4/result_0/generator_0_100000.h5')

gen_mat=[]
it = 0
for k in range(10000):
    
    print("iteration:%d" %(k))
    
    noise = np.random.normal(0,1,(1,100))
    
    gen_relz = generator.predict(noise)[0,:,:,0] # 12 x 12 x 1 ( -1 ~ 1)
    
    gen_relz_converted = ths(gen_relz,5) # 12 x 12 x 1 (0,1,2,3,4)
    gen_mat=[*gen_mat, gen_relz_converted]

gen_net=np.array(gen_mat)
gen_net.shape       

gen_net1=np.array(gen_net).transpose(1,2,0)
gen_net1.shape

savemat('gen_tr_layer1_labels_0.mat', dict([('t01', gen_net1)]))

#a = generator.predict(noise)