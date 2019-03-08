import tensorflow as tf
import numpy as np 
import time
from RiverGAN import RiverGAN
from scipy.io import loadmat, savemat

from keras.optimizers import Adam

# set beta
beta = 1000

# number of input layers(nchannel)
nchannels = 1

# number of z of G(z) - typically 100
latent_dim = 100

# optimizer
optimizer_func = Adam(0.0002, 0.5)
# loss function
loss_d='binary_crossentropy'
loss_g='binary_crossentropy'

# Load training images
net = loadmat('input_%d_img.mat' % (beta))

TI=net['TI']
print('TI.shape = ', (TI.shape))

#TI=np.copy(ti)
#TI = TI.reshape(ti.shape[0],ti.shape[1],ti.shape[2],1)

# Case directory & result directory
case_n=1   # directory of case
case_nn=0  # directory of result in case

params = {'channels':nchannels,'latent_dim':latent_dim,'optimizer_func':optimizer_func,\
'loss_d':loss_d,'loss_g':loss_g, 'case_n':case_n, 'case_nn':case_nn}
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# intialze
myGan = RiverGAN(TI,params)

time_start=time.time()

# im_save_interval: image save interval, model_save_interval: G and D save interval
#myGan.train(epochs=100001, batch_size=32, im_save_interval=50, model_save_interval=5000)
myGan.train(epochs=100001, batch_size=32, im_save_interval=5000, model_save_interval=5000)

time_elapsed=(time.time()-time_start)

print("---total %s seconds---" % time_elapsed)
sess.close()
