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

def ths(data,thsval,minv=-1.0,maxv=1.0):
    '''
    thresholding for nlevels
    '''
    data_tmp = np.copy(data)
    data_tmp[(data >= minv) & (data < thsval)] = 0
    data_tmp[(data >= thsval) & (data <= maxv)] = 255

    return data_tmp

generator=load_model('./generator_0_35000.h5')

gen_mat=[]
it = 0
for k in range(100):
    
    print("iteration:%d" %(k))
    
    noise = np.random.normal(0,1,(1,100))
    
    gen_relz = generator.predict(noise)[0,:,:,0] # 12 x 12 x 1 ( -1 ~ 1)
    gen_relz_converted = ths(gen_relz,0) # 12 x 12 x 1 
    gen_mat=[*gen_mat, gen_relz_converted]
    
gen_net=np.array(gen_mat)
gen_net.shape       

gen_net1=np.array(gen_net).transpose(1,2,0)
gen_net1.shape

savemat('gen_tr_layer1_img_0.mat', dict([('t01', gen_net1)]))
