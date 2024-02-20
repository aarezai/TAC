print('######### import stuff########')

import tensorflow as tf
print("TensorFlow version:", tf.__version__)

print("GPU:")
print(tf.config.list_physical_devices('GPU'))

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, UpSampling2D
from tensorflow.keras import Model
from tensorflow.python.keras import losses

import time
import numpy as np

import os
import scipy.io
import glob



##### PARAMS
print('######### PARAMETERS ########')
print('Batch size')
var_bs = 1 #<-batch size, decreasing batch size to 1 for testing purposes
print(var_bs)
ds_fctr = 1 #<-down sample factor (1=none, 2=half, etc)
print('ds fctr')
print(ds_fctr)
img_size = 512 #<x-y images size = 512/ds_fctr
print('Img size')
print(img_size)
n_epochs = 100#0


print('######### Training Data ########')

#Training data:

max_z_dim = 98

train = np.array([]).reshape(0,img_size,img_size,max_z_dim)
train_labels = np.array([]).reshape(0,img_size,img_size,max_z_dim)

for file in glob.glob('/scratch/arezai/train/*.mat'):
    mat = scipy.io.loadmat(file)
    print(file)
    CT_seg = np.expand_dims(mat['CT_seg'],0)
    CT_vol = np.expand_dims(mat['CT_vol'],0)
    CT_vol = CT_vol.astype(np.float32)
    if np.shape(CT_vol)[3]>max_z_dim:
        CT_seg = CT_seg[:,0:512:ds_fctr,0:512:ds_fctr,5:57]
        CT_vol = CT_vol[:,0:512:ds_fctr,0:512:ds_fctr,5:57]
    elif np.shape(CT_vol)[3]<max_z_dim:
        print(np.shape(CT_seg))
        print(np.shape(CT_vol))
        CT_seg = np.pad(CT_seg[:,0:512:ds_fctr,0:512:ds_fctr,:],((0,0),(0,0),(0,0),(0,max_z_dim-np.shape(CT_vol)[3])))
        CT_vol = np.pad(CT_vol[:,0:512:ds_fctr,0:512:ds_fctr,:],((0,0),(0,0),(0,0),(0,max_z_dim-np.shape(CT_vol)[3])))
        print(np.shape(CT_seg))
        print(np.shape(CT_vol))
    else:
        CT_seg = CT_seg[:,0:512:ds_fctr,0:512:ds_fctr,:]
        CT_vol = CT_vol[:,0:512:ds_fctr,0:512:ds_fctr,:]
    train = np.concatenate((train, CT_vol),0)
    train_labels = np.concatenate((train_labels, CT_seg),0)
    
print(np.shape(train))
print(np.shape(train_labels))


print('######### VAL Data ########')

# VALIDATION
val = np.array([]).reshape(0,img_size,img_size,max_z_dim)
val_labels = np.array([]).reshape(0,img_size,img_size,max_z_dim)

for file in os.listdir('/scratch/arezai/test'):
    mat = scipy.io.loadmat('/scratch/arezai/test/'+file)
    print(file)
    CT_seg = np.expand_dims(mat['CT_seg'],0)
    CT_vol = np.expand_dims(mat['CT_vol'],0)
    CT_vol = CT_vol.astype(np.float32)
    print(np.shape(CT_vol))
    if np.shape(CT_vol)[3]>max_z_dim:
        CT_seg = CT_seg[:,0:512:ds_fctr,0:512:ds_fctr,5:57]
        CT_vol = CT_vol[:,0:512:ds_fctr,0:512:ds_fctr,5:57]
    elif np.shape(CT_vol)[3]<max_z_dim:
        print(np.shape(CT_seg))
        print(np.shape(CT_vol))
        CT_seg = np.pad(CT_seg[:,0:512:ds_fctr,0:512:ds_fctr,:],((0,0),(0,0),(0,0),(0,max_z_dim-np.shape(CT_vol)[3])))
        CT_vol = np.pad(CT_vol[:,0:512:ds_fctr,0:512:ds_fctr,:],((0,0),(0,0),(0,0),(0,max_z_dim-np.shape(CT_vol)[3])))
        print(np.shape(CT_seg))
        print(np.shape(CT_vol))
    else:
        CT_seg = CT_seg[:,0:512:ds_fctr,0:512:ds_fctr,:]
        CT_vol = CT_vol[:,0:512:ds_fctr,0:512:ds_fctr,:]
    val = np.concatenate((val, CT_vol),0)
    val_labels = np.concatenate((val_labels, CT_seg),0)
    
print(np.shape(val))
print(np.shape(val_labels))




print('######### BUILD UNET ########')


######### MOdified for 3D U-Net
def down_block(x, filters, mp_size_z, kernel_size=(3, 3,3), padding="same", strides=1):
    c = tf.keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = tf.keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = tf.keras.layers.MaxPool3D((2, 2, mp_size_z), (2, 2,mp_size_z))(c)
    return c, p

def up_block(x, skip, filters, mp_size_z, kernel_size=(3, 3,3), padding="same", strides=1):
    us = tf.keras.layers.UpSampling3D((2, 2,mp_size_z))(x)
    concat = tf.keras.layers.Concatenate()([us, skip])
    c = tf.keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = tf.keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def bottleneck(x, filters, kernel_size=(3, 3,3), padding="same", strides=1):
    c = tf.keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = tf.keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c



######### MOdified for 3D U-Net
#3D
def UNet():
    f = [1,1,1,1,1]#[2, 4, 6,1,8]
    print('\n Feature List:')
    print(f)
    inputs = tf.keras.layers.Input((img_size, img_size, max_z_dim,1 ))
    
    p0 = inputs
    c1, p1 = down_block(p0, f[0],2) #512 -> 256
    c2, p2 = down_block(p1, f[1],2) #256 -> 128
    c3, p3 = down_block(p2, f[2],1) #128 -> 64
    #c4, p4 = down_block(p3, f[3],1) #64->32
    
    bn = bottleneck(p3, f[4])
    
    #u1 = up_block(bn, c4, f[3],1) #32->64
    u2 = up_block(bn, c3, f[2],1) #64 -> 128
    u3 = up_block(u2, c2, f[1],2) #128 -> 256
    u4 = up_block(u3, c1, f[0],2) #256 -> 512
    
    outputs = tf.keras.layers.Conv3D(1, (1, 1,1), padding="same", activation="sigmoid")(u4)
    model = tf.keras.models.Model(inputs, outputs)
    return model



##########   Alternative DSC loss
# DSC loss
# https://notebook.community/cshallue/models/samples/outreach/blogs/segmentation_blogpost/image_segmentation
def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss










model = UNet()
# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])])

optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss=dice_loss, metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])

model.summary()








###### Train U-Net 
print('######### TRAIN UNET ########')
#model.fit(train, train_labels.astype(np.float32), batch_size = 5, epochs = 200)
model.fit(train, train_labels.astype(np.float32), validation_data=(val,val_labels), batch_size = var_bs, epochs = n_epochs)
#^^ changed to train_labels.astype(np.float32) to use the DSC loss



#Save Model
print('######### SAVE UNET ########')
model.save_weights("UNetW_DescAo_v1_bs"+str(img_size)+'_'+str(var_bs)+".h5")



print('######### all done ########')
