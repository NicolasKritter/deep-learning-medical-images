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
import ImageProcessing

SEED = 42
random.seed = 42
np.random.seed = 42


IMG_WIDTH = 128 #256
IMG_HEIGHT = 128 #256
IMG_CHANNELS = 1 #3
NB_CLASSES = 1
PICKLE_FILE = 'data.pickle'
TRAIN_PATH='data/train/'
TEST_PATH='data/test/'
EXTENSION = '.tif'


def getImagesNMasks(le_path):
    print("Getting & Resizing train images and mask")
    id_list = next(os.walk(le_path))[1]
    images = np.zeros((len(id_list),IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS),dtype=np.uint8)
    labels = np.zeros((len(id_list),IMG_HEIGHT,IMG_WIDTH,1),dtype = np.bool)
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(id_list), total=len(id_list)):
        path = le_path + id_
        print(id_)
        img = cv2.imread(path + '/images/' + id_ + EXTENSION,IMG_CHANNELS)[:,:,:IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        images[n] = normalizeImage(img)

        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
           #print(mask_file)
            mask_ = cv2.imread(path + '/masks/' + mask_file)[:,:,:NB_CLASSES]
            mask_ = resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            mask = ImageProcessing.getFullMask(mask_)# mask_)
        labels[n] = mask
    return images,labels,id_list  #convert images to [0;1]




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
 
train_images,train_mask,train_ids = getImagesNMasks(TRAIN_PATH)
test_images,test_labels,test_ids = getImagesNMasks(TEST_PATH)

def saveData():
    
    save = {
        'train_images':train_images,
        'train_mask':train_mask,
        'test_image':test_images,
        'test_labels':test_labels,
        'IMG_WIDTH':IMG_WIDTH,
        'IMG_HEIGHT':IMG_HEIGHT,
        'IMG_CHANNELS':IMG_CHANNELS,
        'TEST_DATASET_SIZE':len(test_ids),
        'TRAIN_DATASET_SIZE':len(train_ids)
        }
    print('Data saved')
    savePickleData(PICKLE_FILE,save,True)

def getTrainImagesNMasksBatch(num_batch):
    print("Getting & Resizing train images and mask")
    id_list = next(os.walk(TRAIN_PATH))[1]
    images = np.zeros((num_batch,IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS),dtype=np.uint8)
    labels = np.zeros((num_batch,IMG_HEIGHT,IMG_WIDTH,1),dtype = np.bool)
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(id_list), total=len(id_list)):
        path = TRAIN_PATH + id_
        print(id_)
        img = cv2.imread(path + '/images/' + id_ + EXTENSION,IMG_CHANNELS)[:,:,:IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        images[n] = normalizeImage(img)

        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
           #print(mask_file)
            mask_ = cv2.imread(path + '/masks/' + mask_file)[:,:,:NB_CLASSES]
            mask_ = resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            mask =ImageProcessing.getFullMask(mask_)
        labels[n] = mask
    return images,labels
    
def extractImageNMask(path,id_):
    img = cv2.imread(path + '/images/' + id_ + EXTENSION,IMG_CHANNELS)[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    mask_ = cv2.imread(path + '/masks/' + id_)[:,:,:NB_CLASSES]
    mask = resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    mask = ImageProcessing.getFullMask(mask)
    return img,mask
        
def test_labels():
    plt.title("testing")
    ix = random.randint(0, len(train_ids)-1) #len(X_test) - 1 = 64
    test_image = train_mask[ix].astype(float)[:, :, 0]
    imshow(test_image)
    plt.show()


def normalize(image,minC,maxC):
    image= (image - minC) / (maxC - minC)
    return image

def zero_center(image):
    image = image - image.mean()
    return image
    
def normalizeImage(image):
    minC,maxC = image.min(),image.max()
    image=image.astype('float32')
    image=normalize(image,minC,maxC)
    image=zero_center(image)
    return image
#test_labels()
#saveData()