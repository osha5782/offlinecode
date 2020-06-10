#!/usr/bin/env python
# coding: utf-8

# # Model Attempt

# In[1]:


model_attempt = '13.1'


# ### Variables

# ##### Size
# * Image Size = 256
# 
# ##### Augs
# * Augmentations = N
# 
# ##### Batch & Epoch
# * BATCH SIZE = 1
# * Epochs = 100
# * Steps Per Epoch = 85
# 
# ##### Training Hyperparameters
# * Loss Function = Binary CE Loss
# * Optimiser = Adam
# * Learning Rate = 1e-4
# * Momentum = n/a
# 
# ##### Data Generator
# * Using Sarah's (adapted in 8.3)

# ## Basics

# ### Standard Imports

# In[40]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np # linear algebra
from numpy.random import seed
seed(450396580)

from tensorflow import set_random_seed
set_random_seed(450396580)

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import random, h5py


import segmentation_models as sm
from segmentation_models.metrics import *
from segmentation_models import Unet


import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *

from keras import backend as keras

from segmentation_models.losses import DiceLoss, JaccardLoss, BinaryCELoss, CategoricalCELoss, BinaryFocalLoss, CategoricalFocalLoss


from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, EarlyStopping

from sklearn.metrics import roc_curve, roc_auc_score, auc, jaccard_score, accuracy_score
from keras.utils import to_categorical

from albumentations import (
    Compose,
    HorizontalFlip,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,    
    RandomGamma    
)


# ### File Paths

# #### Tester: Train & Val (1x image, 1x mask)

# In[12]:


#train_image_folder = 'SBdata/train/1image'
#train_mask_folder = 'SBdata/train/1mask'

#val_image_folder = 'SBdata/val/1image'
#val_mask_folder = 'SBdata/val/1mask'


# #### Full: Train & Val (all images & masks)

# In[13]:


train_image_folder = 'SBdata/train/image'
train_mask_folder = 'SBdata/train/mask'

val_image_folder = 'SBdata/val/image'
val_mask_folder = 'SBdata/val/mask'


# Full: Test (images & masks)

# In[14]:


test_image_folder = 'SBdata/test/image'
test_mask_folder = 'SBdata/test/mask'


# #### Save Path

# In[15]:


save_path = f'Model_{model_attempt}'


# ### Defined Variables

# In[16]:


image_size = 256
batch_size = 8
epochs = 100


classes = 1
class_dict = {0:'BG', 1:'SB'} 
class_weights = None


# ## Data Generators

# ### Augmentations

# For testing the 1 image, don't worry about augmentations.

# In[17]:


#aug = ()


# In[105]:


aug = Compose([OneOf([
        ElasticTransform(p=1, alpha=200, sigma=200 * 0.05, alpha_affine=200 * 0.03),
        GridDistortion(p=1, border_mode=0, value=5)], p=0.8),
    HorizontalFlip(p=0.5),    
    RandomGamma(p=0.8)])


# ### Training Data Generator - Train & Val

# Define function

# In[101]:


# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import cv2
# import matplotlib.pyplot as plt
# %matplotlib inline
# import random, cv2, h5py


# In[106]:


def train_gen(img_folder, mask_folder, batch_size=batch_size):
    c=0
    n=os.listdir(img_folder)
    random.shuffle(n)
    
    while (True):
        img = np.zeros((batch_size, 256, 256,1)).astype('float')
        mask = np.zeros((batch_size, 256, 256, 1)).astype('float')
        
        for i in range(c, c+batch_size):
            
            train_img = cv2.imread(img_folder+'/'+n[i], cv2.IMREAD_GRAYSCALE)/255.
            train_mask = cv2.imread(mask_folder+'/'+n[i], cv2.IMREAD_GRAYSCALE)/255.
                        
            augmented = aug(image=train_img, mask=train_mask)
            train_img = augmented['image']
            train_mask = augmented['mask']
            
            
            train_img = cv2.resize(train_img, (256, 256))            
            train_img = train_img.reshape([image_size, image_size,1])
            
            img[i-c] = train_img
            
            
            train_mask = cv2.resize(train_mask, (256, 256))
            train_mask = train_mask.reshape([256, 256, 1])
            
            mask[i-c] = train_mask
        
        #increase c by batch size, feeds through next batch of images
        c+=batch_size
        #if we're at the end, reinitialises c again. you need to make it theoretically infinite otherwise it breaks/wigs out. 
        if (c+batch_size>=len(os.listdir(img_folder))):
            c=0
            random.shuffle(n)
        
        # "return"
        yield img, mask


# Call function

# In[107]:


#creating variable that holds the data generator. calls the above code, stores it in the variable train_gen
train_data = train_gen(train_image_folder, train_mask_folder,batch_size=batch_size)
val_data = train_gen(val_image_folder, val_mask_folder,batch_size=batch_size)


# Show examples

# In[78]:


# fig, ax = plt.subplots(1,2, figsize = (16,8))
# ax[0].imshow(x_data[12], cmap='gray')
# ax[1].imshow(y_data[12], cmap='gray')
# plt.show()


# ### Predicting Data Generator - Val & Test

# Define function

# In[79]:


#def pred_gen()


# Call function

# In[80]:


#valpred_data = pred_gen()
#testpred_data = pred_gen()


# ## Build Model

# ### Define Metrics

# In[81]:


# import segmentation_models as sm
# from segmentation_models.metrics import *


# In[82]:


# IoU - Jaccard Score
IOU = sm.metrics.IOUScore()

# Dice Coefficient
FScore = sm.metrics.FScore()


# In[83]:


metrics = [IOU, FScore]


# ### Define Loss Function

# In[84]:


# import numpy as np 
# import os
# import skimage.io as io
# import skimage.transform as trans
# import numpy as np
# from keras.models import *
# from keras.layers import *
# from keras.optimizers import *
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from keras import backend as keras


# In[85]:


# from segmentation_models.losses import DiceLoss, JaccardLoss, BinaryCELoss, CategoricalCELoss, BinaryFocalLoss, CategoricalFocalLoss


# In[86]:


#loss = sm.losses.JaccardLoss(class_weights=class_weights, class_indexes=None, per_image=False, smooth=1e-05)
loss = sm.losses.DiceLoss()
#loss = sm.losses.BinaryCELoss()
#loss = sm.losses.CategoricalCELoss(class_weights=class_weights, class_indexes=None)
#loss = sm.losses.BinaryFocalLoss(alpha=0.25, gamma=2.0)
#loss = sm.losses.CategoricalFocalLoss(alpha=0.25, gamma=2.0, class_indexes=None)


# ### Define Optimiser

# In[87]:


# from keras.optimizers import *


# In[88]:


lr = 1e-4
opt = Adam(learning_rate=lr)


# ### Create Model

# #### Model Structure

# Include a small summary of things like activation function, weights, etc

# In[89]:


# import segmentation_models as sm
# from segmentation_models import Unet


# #### Name The Model

# In[90]:


# encoder weights will change for transfer learning. Instead of none, will have something different. compare later. Softmax will change - research this in segmentation_models
model = Unet(backbone_name='vgg16', encoder_weights=None, classes=classes, activation = 'sigmoid', encoder_freeze = False, input_shape=(image_size, image_size, 1))


# #### Model Summary

# In[91]:


model.summary()


# ### Compile Model

# In[92]:


model.compile(opt, loss=loss, metrics=metrics)


# ## Initiate Training

# ### Training Inputs

# #### Callbacks

# Paths for Callbacks

# In[93]:


weights_path = f'Weights/{save_path}.h5'
traininglog_path = f'TrainingLog/{save_path}.out'


# Callback Functions

# In[94]:


# from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, EarlyStopping


# In[95]:


weight_saver = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True)

csv_logger = CSVLogger(f'TrainingLog/{save_path}.out', append=True, separator=';')

earlystopping = EarlyStopping(monitor="val_loss", mode="min", patience=10)


# In[96]:


callbacks_list = [weight_saver, csv_logger, earlystopping]


# #### Steps Per Epoch

# In[97]:


no_training_images = len(os.listdir(train_image_folder))
no_val_images = len(os.listdir(val_image_folder))


# In[98]:


train_steps_per_epoch = (3*no_training_images//batch_size)
val_steps = (no_val_images//batch_size)


# ### Begin Training

# In[109]:


results = model.fit_generator(train_data, epochs=epochs, 
                          steps_per_epoch = (train_steps_per_epoch),
                          validation_data=val_data,
                          validation_steps=val_steps, 
                             verbose=1, 
                             callbacks = callbacks_list)


# ## Training Results

# #### Read Training Results

# In[110]:


train_results = pd.read_csv(traininglog_path, sep=';')


# #### Define Plot Results

# In[112]:


def plot_results(train_results, save=False, name=None):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14,5))
    
    # Plot loss values
    ax1.plot(train_results['loss'])
    ax1.plot(train_results['val_loss'])
    ax1.title.set_text('Model Loss')
    ax1.set_ylabel('Dice Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train','Validation'], loc='upper right')
    
    # Plot metric results
    ax2.plot(train_results['iou_score'])
    ax2.plot(train_results['val_iou_score'])
    ax2.title.set_text('Model Accuracy (IoU)')
    ax2.set_ylabel('Jaccard Score')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot metric results
    ax3.plot(train_results['f1-score'])
    ax3.plot(train_results['val_f1-score'])
    ax3.title.set_text('Model Accuracy (Dice Coeff)')
    ax3.set_ylabel('Dice Coefficient')
    ax3.set_xlabel('Epoch')
    ax3.legend(['Train', 'Validation'], loc='lower right')
    
    fig.tight_layout()
    fig.show()
    
    # If save is true, save the figure in figures folder with the name defined in function, otherwise just show the image
    if save==False:
        return
    else:
        fig.savefig(f'Figures/{name}.png')
        
    return


# #### Plot Training Results

# In[113]:


plot_results(train_results, save=True, name=save_path)


# ## Val Predictions

# If loading pre-trained weights

# In[134]:


#model.load_weights('Weights/Model_8.5.2.h5')


# In[135]:


# from sklearn.metrics import roc_curve, roc_auc_score, auc, jaccard_score, accuracy_score
# from keras.utils import to_categorical


# ### Define and Generate Val/Test Data

# In[154]:


def pred_gen(img_folder, mask_folder, batch_size):
    c=0
    n=os.listdir(img_folder)
    n.sort() #always evaluate in the same order 
    #remove shuffling
    while (True):
        img = np.zeros((batch_size, 256, 256,1)).astype('float')
        mask = np.zeros((batch_size, 256, 256, 1)).astype('float')
        
        for i in range(c, c+batch_size):         
            
            train_img = cv2.imread(img_folder+'/'+n[i], cv2.IMREAD_GRAYSCALE)/255.
            train_img = cv2.resize(train_img, (256, 256))
            
            train_img = train_img.reshape([image_size, image_size,1])
            
            img[i-c] = train_img
            
            train_mask = cv2.imread(mask_folder+'/'+n[i], cv2.IMREAD_GRAYSCALE)/255.
            train_mask = cv2.resize(train_mask, (256, 256))
            train_mask = train_mask.reshape([256, 256, 1])
            
            mask[i-c] = train_mask
            
        c+=batch_size
        if (c+batch_size>=len(os.listdir(img_folder))):
            c=0
        
        yield img, mask


# In[155]:


# Generate val predictions
val_pred_generator = pred_gen(val_image_folder, val_mask_folder,batch_size=19)
test_pred_generator = pred_gen(test_image_folder, test_mask_folder,batch_size=26)


# ### Define Jaccard Score

# In[157]:


# might change for binary - reserach and adapt. 4 is the classes, so would likely change that
def get_jaccard(image_folder, mask_folder, generator, batch_size):
    image_names = os.listdir(image_folder)
    image_names.sort()
    
    predictions = model.predict_generator(generator, steps=1)
    
    ytrue = np.zeros((batch_size, image_size, image_size, 1)).astype('float')

    for i in range(batch_size):
        target = cv2.imread(os.path.join(mask_folder, image_names[i]), cv2.IMREAD_GRAYSCALE)
        target = cv2.resize(target, (image_size, image_size))
        target = target.reshape([image_size, image_size, 1])
        
        ytrue[i] = target
    
    ypred = predictions>0.5
    
    scores = jaccard_score(ytrue.ravel(), ypred.ravel(), average='weighted')
    
    return print(f'SB: {scores[0]}')


# In[158]:


# Val & Test Jaccard Score
Val_IoU2 = get_jaccard(val_image_folder, val_mask_folder, val_pred_generator, batch_size=19)


# In[116]:


# might change for binary - reserach and adapt. 4 is the classes, so would likely change that
def get_jaccard(image_folder, mask_folder, generator, batch_size):
    image_names = os.listdir(image_folder)
    image_names.sort()
    
    predictions = model.predict_generator(generator, steps=1)
    
    ytrue = np.zeros((batch_size, image_size, image_size, 1)).astype('float')

    for i in range(batch_size):
        target = cv2.imread(os.path.join(mask_folder, image_names[i]), cv2.IMREAD_GRAYSCALE)
        target = cv2.resize(target, (image_size, image_size))
        target = target.reshape([image_size, image_size, 1])
        
        ytrue[i] = target
    
    ypred = predictions>0.5
    
    scores = jaccard_score(ytrue.ravel(), ypred.ravel(), average=None)
    
    return print(f'SB: {scores[0]}')


# #### Jaccard Score

# In[117]:


# Val & Test Jaccard Score
Val_IoU = get_jaccard(val_image_folder, val_mask_folder, val_pred_generator, batch_size=19)


# In[118]:


Test_IoU = get_jaccard(test_image_folder, test_mask_folder, test_pred_generator, batch_size=26)


# ### Predict Mask With Image

# In[127]:


# # Predict val or test images
image_names = os.listdir(val_image_folder)
image_names.sort() 
predictions = model.predict_generator(val_pred_generator, steps=1)
ypred_new = predictions>0.5


# In[128]:


#predictions.shape


# In[129]:


#print(predictions)


# In[130]:


#ypred_new = predictions > 0.5


# In[131]:


#print(ypred_new)


# In[132]:


#np.count_nonzero(ypred_new)


# #### Define Prediction Plot Function

# In[133]:


def plot_pred(index, predictions = ypred_new, save=False):
    
    img = cv2.imread(os.path.join(val_image_folder,image_names[index]), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (image_size, image_size))   

    target = cv2.imread(os.path.join(val_mask_folder, image_names[index]), cv2.IMREAD_GRAYSCALE)
    target = cv2.resize(target, (image_size, image_size))

    scores = jaccard_score(target.ravel(), ypred_new[index].ravel(), average=None)

    plt.figure(figsize=(16,18))
    
    plt.subplot(1,3,1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.annotate('CT Image',(image_size*0.1, image_size*0.95), color='white')    
    
    plt.subplot(1,3,2)
    plt.imshow(img, cmap='gray')
    plt.imshow(target, cmap = 'inferno', alpha=0.4)
    plt.axis('off')
    plt.annotate('Ground Truth',(image_size*0.1, image_size*0.95), color='white')    
    
    plt.subplot(1,3,3)
    plt.imshow(img, cmap='gray')
    plt.imshow(ypred_new[index,:,:,0], cmap = 'inferno', alpha=0.4)
    plt.axis('off')
    plt.annotate(f'IoU: {round((scores[0]*100),4)}',(image_size*0.1, image_size*0.95), color='white')
    
    if save == True:
        plt.savefig(f'Figures/{save_path}_{image_names[index]}')
    
    plt.show()


# In[134]:


plot_pred(0, ypred_new, save=True)
plot_pred(1, ypred_new, save=True)
plot_pred(2, ypred_new, save=True)
plot_pred(3, ypred_new, save=True)
plot_pred(4, ypred_new, save=True)
plot_pred(5, ypred_new, save=True)
plot_pred(6, ypred_new, save=True)
plot_pred(7, ypred_new, save=True)
plot_pred(8, ypred_new, save=True)
plot_pred(9, ypred_new, save=True)
plot_pred(10, ypred_new, save=True)
plot_pred(11, ypred_new, save=True)
plot_pred(12, ypred_new, save=True)
plot_pred(13, ypred_new, save=True)
plot_pred(14, ypred_new, save=True)
plot_pred(15, ypred_new, save=True)
plot_pred(16, ypred_new, save=True)
plot_pred(17, ypred_new, save=True)
plot_pred(18, ypred_new, save=True)


# In[135]:


# Test images
image_names = os.listdir(test_image_folder)
image_names.sort() 
predictions = model.predict_generator(test_pred_generator, steps=1)
ypred_new = predictions>0.5


# In[136]:


def plot_test_pred(index, predictions = ypred_new, save=False):
    
    img = cv2.imread(os.path.join(test_image_folder,image_names[index]), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (image_size, image_size))   

    target = cv2.imread(os.path.join(test_mask_folder, image_names[index]), cv2.IMREAD_GRAYSCALE)
    target = cv2.resize(target, (image_size, image_size))

    scores = jaccard_score(target.ravel(), ypred_new[index].ravel(), average=None)

    plt.figure(figsize=(16,18))
    
    plt.subplot(1,3,1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.annotate('CT Image',(image_size*0.1, image_size*0.95), color='white')    
    
    plt.subplot(1,3,2)
    plt.imshow(img, cmap='gray')
    plt.imshow(target, cmap = 'inferno', alpha=0.4)
    plt.axis('off')
    plt.annotate('Ground Truth',(image_size*0.1, image_size*0.95), color='white')    
    
    plt.subplot(1,3,3)
    plt.imshow(img, cmap='gray')
    plt.imshow(ypred_new[index,:,:,0], cmap = 'inferno', alpha=0.4)
    plt.axis('off')
    plt.annotate(f'IoU: {round((scores[0]*100),4)}',(image_size*0.1, image_size*0.95), color='white')
    
    if save == True:
        plt.savefig(f'Figures/{save_path}_{image_names[index]}')
    
    plt.show()


# In[137]:


plot_test_pred(0, ypred_new, save=True)
plot_test_pred(1, ypred_new, save=True)
plot_test_pred(2, ypred_new, save=True)
plot_test_pred(3, ypred_new, save=True)
plot_test_pred(4, ypred_new, save=True)
plot_test_pred(5, ypred_new, save=True)
plot_test_pred(6, ypred_new, save=True)
plot_test_pred(7, ypred_new, save=True)
plot_test_pred(8, ypred_new, save=True)
plot_test_pred(9, ypred_new, save=True)
plot_test_pred(10, ypred_new, save=True)
plot_test_pred(11, ypred_new, save=True)
plot_test_pred(12, ypred_new, save=True)
plot_test_pred(13, ypred_new, save=True)
plot_test_pred(14, ypred_new, save=True)
plot_test_pred(15, ypred_new, save=True)
plot_test_pred(16, ypred_new, save=True)
plot_test_pred(17, ypred_new, save=True)
plot_test_pred(18, ypred_new, save=True)
plot_test_pred(19, ypred_new, save=True)
plot_test_pred(20, ypred_new, save=True)
plot_test_pred(21, ypred_new, save=True)
plot_test_pred(22, ypred_new, save=True)
plot_test_pred(23, ypred_new, save=True)
plot_test_pred(24, ypred_new, save=True)
plot_test_pred(25, ypred_new, save=True)


# In[ ]:




