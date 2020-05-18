#!/usr/bin/env python
# coding: utf-8

# # 0. Model Overview
# 
# ## General Notes
# Customising the Lung CT Segmentation a bit more to have correct folder paths and other things. 
# 108 images in total (from 1 case I believe), train/val split of 0.5 as defined in the code 
# I believe that the train-test split is actually "train-val" split and that there isn't testing on this yet. 
# I think moving forward that testing would pull from a separate folder and defined and created later on. 
# 
# Code source:
# https://www.kaggle.com/toregil/a-lung-u-net-in-keras
# 
# Things added in:
# * Callbacks: EarlyStopping, ReduceLROnPlatau, CSVLogger
# 
# 
# ## Hyperparameters/Variables
# * Image size: 512 x 512
# * Batch size: 8
# * Epochs: 50
# * Test/val split: 0.5
# * Activation function: Sigmoid
# * Data augmentation: done through Keras ImageDataGenerator
# * Loss function: binary_crossentropy
# * Optimizer: Adam
#     * Learning rate: 0.0002 (2e-4)
# * Metric: dice_coef
# * Callbacks: ModelCheckpoint, LearningRateScheduler

# # 1. Model

# ### i. Import Packages

# In[26]:


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split


# In[27]:


from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger


# ### ii. Load and Resize the images

# Image Path; Image Size; Seed

# In[28]:


IMAGE_LIB = 'KLinput/2d_images_SB_108/'
MASK_LIB = 'KLinput/2d_masks_SB_108/'
IMG_HEIGHT, IMG_WIDTH = 512, 512
SEED=42
BATCH_SIZE = 8


# Total Images (just names, no .png)

# In[29]:


all_images = [x for x in sorted(os.listdir(IMAGE_LIB)) if x[-4:] == '.png']

print('Total number of images:', len(all_images))


# Read Images and Masks

# In[30]:


x_data = np.empty((len(all_images), IMG_HEIGHT, IMG_WIDTH), dtype='float32')
for i, name in enumerate(all_images):
    im = cv2.imread(IMAGE_LIB + name, cv2.IMREAD_GRAYSCALE).astype("int16").astype('float32')
    im = cv2.resize(im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    x_data[i] = im

y_data = np.empty((len(all_images), IMG_HEIGHT, IMG_WIDTH), dtype='float32')
for i, name in enumerate(all_images):
    im = cv2.imread(MASK_LIB + name, cv2.IMREAD_GRAYSCALE).astype('float32')/255.
    im = cv2.resize(im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
    y_data[i] = im


# Show example image and mask

# In[31]:


fig, ax = plt.subplots(1,2, figsize = (8,4))
ax[0].imshow(x_data[0], cmap='gray')
ax[1].imshow(y_data[0], cmap='gray')
plt.show()


# Split images and masks into train and val

# In[32]:


x_data = x_data[:,:,:,np.newaxis]
y_data = y_data[:,:,:,np.newaxis]
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size = 0.5)


# ### iii. Define and train model

# Metrics

# In[33]:


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())


# In[42]:


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


# In[34]:


metrics = [dice_coef]


# Model Architecture

# In[35]:


input_layer = Input(shape=x_train.shape[1:])
c1 = Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same')(input_layer)
l = MaxPool2D(strides=(2,2))(c1)
c2 = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same')(l)
l = MaxPool2D(strides=(2,2))(c2)
c3 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(l)
l = MaxPool2D(strides=(2,2))(c3)
c4 = Conv2D(filters=32, kernel_size=(1,1), activation='relu', padding='same')(l)
l = concatenate([UpSampling2D(size=(2,2))(c4), c3], axis=-1)
l = Conv2D(filters=32, kernel_size=(2,2), activation='relu', padding='same')(l)
l = concatenate([UpSampling2D(size=(2,2))(l), c2], axis=-1)
l = Conv2D(filters=24, kernel_size=(2,2), activation='relu', padding='same')(l)
l = concatenate([UpSampling2D(size=(2,2))(l), c1], axis=-1)
l = Conv2D(filters=16, kernel_size=(2,2), activation='relu', padding='same')(l)
l = Conv2D(filters=64, kernel_size=(1,1), activation='relu')(l)
l = Dropout(0.5)(l)
output_layer = Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid')(l)
                                                         
model = Model(input_layer, output_layer)


# In[36]:


model.summary()


# Data Augmentation (using ImageDataGenerator)

# In[37]:


def my_generator(x_train, y_train, batch_size):
    data_generator = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            zoom_range=0.1).flow(x_train, x_train, batch_size, seed=SEED)
    mask_generator = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            zoom_range=0.1).flow(y_train, y_train, batch_size, seed=SEED)
    while True:
        x_batch, _ = data_generator.next()
        y_batch, _ = mask_generator.next()
        yield x_batch, y_batch


# In[38]:


# By using the same RNG seed in both calls to ImageDataGenerator, 
# we should get images and masks that correspond to each other. 
# Let's check this, to be safe.


# In[41]:


image_batch, mask_batch = next(my_generator(x_train, y_train, 8))
fix, ax = plt.subplots(8,2, figsize=(8,20))
for i in range(8):
    ax[i,0].imshow(image_batch[i,:,:,0])
    ax[i,1].imshow(mask_batch[i,:,:,0])
plt.show()


# Optimizer with Learning Rate

# In[16]:


opt = Adam(2e-4)


# Loss Function

# In[17]:


loss = 'binary_crossentropy'


# Compile Model

# In[15]:


model.compile(optimizer=opt, loss=loss, metrics=metrics)


# CallBacks

# In[16]:


weight_saver = ModelCheckpoint('lung.h5', monitor='val_dice_coef', 
                                              save_best_only=True, save_weights_only=True)
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.8 ** x)


# In[25]:


# --------------------------------------- #
# --------------------------------------- #
        # VARIABLES FOR NAMING

size = '512'
batchsize = '8'
epochs = '50'
split = '0.5'
metric_names = 'DC'
Unet_act = 'sigmoid'
augmentations = 'y'
opt_name = 'Adam'
lr = '2e-4'
loss_name = 'BinaryCE'

# --------------------------------------- #
# --------------------------------------- #


# In[17]:


# changed steps_per_epoch from 200 to len(x_train)//8 -- number of training images divided by batch size as this uses
# all of the data points, one batch size worth at a time

hist = model.fit_generator(my_generator(x_train, y_train, batch_size),
                           steps_per_epoch = (len(x_train)//8),
                           validation_data = (x_val, y_val),
                           epochs=50, verbose=1,
                           callbacks = [weight_saver, annealer])


# ### Evaluate

# In[18]:


model.load_weights('lung.h5')


# In[19]:


plt.plot(hist.history['loss'], color='b')
plt.plot(hist.history['val_loss'], color='r')
plt.show()
plt.plot(hist.history['dice_coef'], color='b')
plt.plot(hist.history['val_dice_coef'], color='r')
plt.show()


# In[20]:


plt.imshow(model.predict(x_train[0].reshape(1,IMG_HEIGHT, IMG_WIDTH, 1))[0,:,:,0], cmap='gray')


# In[21]:


y_hat = model.predict(x_val)
fig, ax = plt.subplots(1,3,figsize=(12,6))
ax[0].imshow(x_val[0,:,:,0], cmap='gray')
ax[1].imshow(y_val[0,:,:,0])
ax[2].imshow(y_hat[0,:,:,0])


# In[ ]:




