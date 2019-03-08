"""
RiverGANs 
more description will be provided soon
Sung Eun Kim, Jonghyun Harry Lee
"""

import tensorflow as tf

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

#from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.optimizers import SGD, RMSprop
#import keras.backend; keras.backend.clear_session()
#from keras.layers.convolutional import MaxPooling2D

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os 
import time

class RiverGAN:
    
    def __init__(self, TI, params):
        '''intialization
        '''
        # Input shape
        self.TI = TI
        self.img_rows = TI.shape[1]
        self.img_cols = TI.shape[2]
        self.channels = params['channels']
        
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = params['latent_dim']
        self.g_ini_rows=int(self.img_rows/4) 
        self.g_ini_cols=int(self.img_cols/4)
        
        self.optimizer = params['optimizer_func']
        self.loss_d = params['loss_d']
        self.loss_g = params['loss_g']

        self.case_n = params['case_n'] 
        self.case_nn = params['case_nn']

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.loss_d,optimizer=self.optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates networks
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model, we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        self.combined = Model(z, validity)
        self.combined.compile(loss=self.loss_g, optimizer=self.optimizer)

    def build_generator(self):
        '''
        build generator G
        '''
        model = Sequential()

        model.add(Dense(128 * (self.g_ini_rows) * (self.g_ini_cols), activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((self.g_ini_rows, self.g_ini_cols, 128)))
        
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        '''
        build discriminator D
        '''
        
        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        #model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        #model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        #model.add(BatchNormalization(momentum=0.8))
        #model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.25))
        
        #model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        #model.add(BatchNormalization(momentum=0.8))
        #model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)
    

    def train(self, epochs, batch_size=128, im_save_interval=50, model_save_interval=20):
        '''training
        '''
        # Load the dataset
        X_train=self.TI

        if X_train.ndim != 4:
            raise ValueError('train data.ndim = %d instead of 4, your training data has a shape of (# of training samples, img size x, img size y, nchannels)' % (X_train.ndim))
        # Rescale -1 to 1
        
        for i in range(X_train.shape[3]):
            unique_vals = np.unique(X_train[:,:,:,i])
            X_train[:,:,:,i] = 2.*(X_train[:,:,:,i]- (unique_vals.max() + unique_vals.min())/2.)/(unique_vals.max() - unique_vals.min()) 
            print(np.unique(X_train[:,:,:,i]))

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        d_loss_history=np.zeros((epochs,6))
        g_loss_history=np.zeros((epochs,1))

        case_n = self.case_n 
        case_nn = self.case_nn
        # result files
        run_time='./Case_%d/result_%d/training_time_%d.csv' %(self.case_n, self.case_nn, self.case_nn) 
        name_gn='./Case_%d/result_%d/generator_%d' %(case_n, case_nn, case_nn) #self.generator.save("generator.h5")
        name_dn='./Case_%d/result_%d/discriminator_%d' %(case_n, case_nn, case_nn) #self.discriminator.save("discriminator.h5")
        name_gn2='./Case_%d/result_%d/G_loss_%d' %(case_n, case_nn, case_nn) #np.savetxt(name_gn2, g_loss, delimiter=',')
        name_dn2='./Case_%d/result_%d/D_loss_%d' %(case_n, case_nn, case_nn) #np.savetxt(name_dn2, d_loss, delimiter=',')
        
        os.makedirs(os.path.dirname(run_time), exist_ok=True)        
        rt=open(run_time,'w')

        time_start=time.time()
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]                          

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            d_loss_history[epoch]=(d_loss_real[0],d_loss_fake[0], d_loss[0], d_loss_real[1],d_loss_fake[1], d_loss[1])
            

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)
            
            g_loss_history[epoch]=g_loss

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            #if epoch % im_save_interval == 0:
            #    self.save_imgs(epoch)
                
            if (epoch > 0) and (epoch % model_save_interval == 0):
                self.generator.save(name_gn+'_%d.h5' % epoch)
                self.discriminator.save(name_dn+'_%d.h5' % epoch)
                
            if (epoch > 0) and (epoch % 5000) ==0:
                time_elapsed=(time.time()-time_start)
                rt.write(str(time_elapsed)+'\n')

        rt.close()
        
        np.savetxt(name_dn2+'_%d.csv' % (epoch), d_loss_history, delimiter=',')
        np.savetxt(name_gn2+'_%d.csv' % (epoch), g_loss_history, delimiter=',')                    
       
if __name__ == '__main__':

    import time
    from RiverGAN import RiverGAN
    
    # number of input layers(channel)
    channels = 3
    # number of z of G(z)
    latent_dim = 100
    # optimizer
    optimizer_func = Adam(0.0002, 0.5)
    # loss function
    loss_d='binary_crossentropy'
    loss_g='binary_crossentropy'

    # Load training images
    net= loadmat('input_3layer_lru.mat', squeeze_me=True)
    #net.keys()

    ti=net['ti']
    print(ti.shape)

    TI=np.copy(ti)

    # Case directory & result directory
    case_n=0   # directory of case
    case_nn=3  # directory of result in case

    
    params = {'channels':channels,'latent_dim':latent_dim,'optimizer_func':optimizer_func,'loss_d':loss_d,'loss_g':loss_g, 'case_n':case_n, 'case_nn':case_nn}
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

    # test relz1
    #noise = np.random.normal(0,1,(1,100))
    #relz1 = myGan.generator.predict(noise)[0,:,:,:] # 12 x 12 x 2 ( -1 ~ 1)

    #generator.predict(noise)[0,:,:,:] # 12 x 12 x 2 ( -1 ~ 1)
    #convert to 0 ~ 2
    #relz1 = 1.0*relz1 + 1.0

# fig, ax = plt.subplots(nrows = 1, ncols = 2, sharey=True)
# im0 = ax[0].imshow(relz1[:,:,0])
# plt.colorbar(im0,cax=ax[0])
# im1 = ax[1].imshow(relz1[:,:,1])
# plt.colorbar(im1,cax=ax[1])
# plt.show()


