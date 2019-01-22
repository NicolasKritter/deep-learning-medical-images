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
import matplotlib.pyplot as plt
from skimage.io import  imshow
import cv2
import ImageProcessing

#1336*888
#FACTOR = 2.6
IMG_WIDTH = 512 #512 512
IMG_HEIGHT = 512 #512 340
IMG_CHANNELS = 3 #3
MASK_CHANNELS = 3
NB_CLASSES = 3
PICKLE_FILE = 'data.pickle'
TRAIN_PATH='data/train/'
TEST_PATH='data/test/'
EXTENSION = '.tif'
#TYPE = '_GT'
TYPE='_segmented'
#print(next(os.walk(TRAIN_PATH))[1])
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

#Rotate the mask & image randomly 
def randomizeAngle(image,mask):
    r = [0,90,180,270]
    rot = np.random.choice(r)
    image = ImageProcessing.rotation(image,rot)
    mask = ImageProcessing.rotation(mask,rot)
    return image,mask
    
#TODO ajouter flip
def generateTransfo(image,mask):
    if random.randint(0, 5)>=1:
        image,mask = randomizeAngle(image,mask)
    
    return image,mask

    
def extractImageNMask(path,id_):
    img = cv2.imread(path + '/images/' + id_ + EXTENSION)[:,:,:IMG_CHANNELS]
    mask = cv2.imread(path + '/masks/' + id_+TYPE+EXTENSION)[:,:,:MASK_CHANNELS]
    img = ImageProcessing.crop(img,20)
    mask = ImageProcessing.crop(mask,20)
    #img,mask=generateTransfo(img,mask)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC)
    img = normalizeImage(img)
    mask = ImageProcessing.getFullMaskFromImg(mask)
    mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_NEAREST)
    return img,mask
    

def getImagesNMasks(le_path):
    print("Getting & Resizing train images and mask")
    id_list = next(os.walk(le_path))[1]
    size  = len(id_list)
    images = np.zeros((size,IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS),dtype="float32")
    labels = np.zeros((size,IMG_WIDTH,IMG_HEIGHT),dtype="uint8")#uint8
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(id_list), total=len(id_list)):
        #print(id_)
        path = le_path + id_
        images[n],labels[n] = extractImageNMask(path,id_)

    return images,labels,size




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

def getTrainBatch():
    return getImagesNMasks(TRAIN_PATH)
 
if __name__ == '__main__':
    train_images,train_mask,train_size = getImagesNMasks(TRAIN_PATH)
    train_images,train_mask = ImageProcessing.formatData(train_images,train_mask,3)

    test_images,test_labels,test_size = getImagesNMasks(TEST_PATH)
    test_images,test_labels = ImageProcessing.formatData(test_images,test_labels,3)

def saveData():
    
    save = {
        'train_images':train_images,
        'train_mask':train_mask,
        'test_image':test_images,
        'test_labels':test_labels,
        'IMG_WIDTH':IMG_WIDTH,
        'IMG_HEIGHT':IMG_HEIGHT,
        'IMG_CHANNELS':IMG_CHANNELS,
        'TEST_DATASET_SIZE':test_size,
        'TRAIN_DATASET_SIZE':train_size
        }
    savePickleData(PICKLE_FILE,save,True)
    print('Data saved')

def getTrainImagesNMasksBatch(num_batch):
    print("Getting & Resizing train images and mask")
    id_list = next(os.walk(TRAIN_PATH))[1]
    images = np.zeros((len(id_list),IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS),dtype="float32")
    labels = np.zeros((len(id_list),IMG_HEIGHT,IMG_WIDTH,NB_CLASSES),dtype="uint8")
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(id_list), total=len(id_list)):
        path = TRAIN_PATH + id_
        #print(id_)
        images[n],labels[n] = extractImageNMask(path,id_)
    return images,labels
    

        
def test_data():
    plt.title("testing")
    ix = random.randint(0, train_size-1) #len(X_test) - 1 = 64
    test_image = train_mask[ix].astype(float)[:, :]
    imshow(test_image)
    plt.show()
    test_image = train_images[ix].astype(float)[:, :, :IMG_CHANNELS]
    imshow(test_image)
    plt.show()
    ix2 = random.randint(0, test_size-1)
    test_image = test_images[ix2].astype(float)[:, :, :IMG_CHANNELS]
    imshow(test_image)
    plt.show()
    test_image = test_labels[ix2].astype(float)[:, :, :IMG_CHANNELS]
    imshow(test_image)
    plt.show()

if __name__ == '__main__':
    #test_data()
    saveData()