# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 16:19:25 2018

@author: nicolas
"""

import os
import sys
import numpy as np
import random
from six.moves import cPickle as pickle
from tqdm import tqdm
from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage.io import  imshow
import cv2

SEED = 42
random.seed = 42
np.random.seed = 42


IMG_WIDTH = 128 #256
IMG_HEIGHT = 128 #236
IMG_CHANNELS = 1 #3
PICKLE_FILE = 'data.pickle'
TRAIN_PATH='data/train/'
TEST_PATH='data/test/'
EXTENSION = '.tif'
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]




def getTrainImagesNMasks():
    print("Getting & Resizing train images and mask")
    images = np.zeros((len(train_ids),IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS),dtype=np.uint8)
    labels = np.zeros((len(train_ids),IMG_HEIGHT,IMG_WIDTH,1),dtype = np.bool)
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_
        img = cv2.imread(path + '/images/' + id_ + EXTENSION,IMG_CHANNELS)[:,:,:IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        images[n] = img
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
           #print(mask_file)
            mask_ = cv2.imread(path + '/masks/' + mask_file)[:,:,:1]
            mask_ = resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            mask = mask_#np.maximum(mask, mask_)
        labels[n] = mask
    return images,labels

train_images,train_mask = getTrainImagesNMasks()

def getTestImages():
    print("Getting & Resizing test images and mask")
    test_images = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    #sizes_test = []
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TEST_PATH + id_
        img = cv2.imread(path + '/images/' + id_ + EXTENSION,IMG_CHANNELS)[:,:,:IMG_CHANNELS]
        #sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        test_images[n] = img
    return test_images#,sizes_test


test_images = getTestImages()#,size_test

train_images = train_images/255 #convert images to [0;1]

def savePickleData(pickle_file,save,force:False):
    if force or not os.path.exists(pickle_file):
        print("Saving dataset")
        try:
            f = open(pickle_file,'wb')
            pickle.dump(save,f,pickle.HIGHEST_PROTOCOL)
            f.close()
        except Exception as e:
            print('Unable to save data to',pickle_file,':',e)
            raise
    return os.stat(pickle_file)

save = {
    'train_images':train_images,
    'train_mask':train_mask,
    'test_image':test_images,
    'IMG_WIDTH':IMG_WIDTH,
    'IMG_HEIGHT':IMG_HEIGHT,
    'IMG_CHANNELS':IMG_CHANNELS,
    'TEST_DATASET_SIZE':len(test_ids),
    'TRAIN_DATASET_SIZE':len(train_ids)
    }
savePickleData(PICKLE_FILE,save,True)



def test_labels():
    plt.title("testing")
    ix = random.randint(0, len(train_ids)-1) #len(X_test) - 1 = 64
    test_image = train_mask[ix].astype(float)[:, :, 0]
    imshow(test_image)
    plt.show()
test_labels()