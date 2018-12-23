# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 17:37:17 2018

@author: nicolas
"""
import numpy as np

from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage.io import  imshow
import tensorflow as tf
from keras.utils import np_utils
import cv2

#OpenCV: BGR
LOWER_RED = np.array([27,0,130])#136,0,27
UPPER_RED = np.array([29,1,136])

LOWER_YELLOW = np.array([0,254,254])
UPPER_YELLOW = np.array([0,255,255])

#PATH
#TEST_PATH= 'test/1.tif'
TEST_PATH= 'data/train/HPF4-483S1-4B-6500x-4/masks/HPF4-483S1-4B-6500x-4_segmented.tif'

#hot encoding
HOT_COEUR = np.array([1,0,0])#0,1 #yellow
HOT_GAINE = np.array([2,0,0])

def getCleanMask(img,lower_bound=np.array([0,0,0]),upper_bound=np.array([255,255,255])):
    mask = cv2.inRange(img, lower_bound, upper_bound)
    res = cv2.bitwise_and(img,img, mask= mask)
    kernel = np.ones((6,6),np.uint8)
    closing = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    return opening
  
def mergeMasks(masks):
    m1 = masks[0]
    for i in range(1,len(masks)):
        np.putmask(m1,masks[i]>0,masks[i].max())
    return m1
    
def getRedMask(img):
    red_mask = getCleanMask(img,LOWER_RED,UPPER_RED)
    red_mask = cv2.cvtColor(red_mask, cv2.COLOR_BGR2GRAY)
    np.putmask(red_mask,red_mask>0,1)
    return  red_mask
def getYellowMask(img):
    yellow_mask = getCleanMask(img,LOWER_YELLOW,UPPER_YELLOW)
    yellow_mask = cv2.cvtColor(yellow_mask, cv2.COLOR_BGR2GRAY)
    np.putmask(yellow_mask,yellow_mask>0,2)
    return yellow_mask   

def getFullMaskFromImg(img,toOne=True):
    m1 = getRedMask(img)
    m2 = getYellowMask(img)
    img = mergeMasks([m1,m2])
    return img


def rotation(img,angle):
    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    return cv2.warpAffine(img,M,(cols,rows))
    
def formatData(X,y,numclass):
     """format list data (X) and tag(y) for cnn"""
     y_tab= np.asarray(y)
     y_cat=np.zeros((y_tab.shape[0],y_tab.shape[1],y_tab.shape[2],numclass),np.uint8)
     print(y_tab.max())
     x_tab= np.asarray(X)

     for i in range (y_tab.shape[0]):
         for j in range (0,y_tab.shape[1]):
#            print i,j,y_test[i][j]
             y_cat[i][j] = np_utils.to_categorical(y_tab[i][j], numclass)
     return  x_tab, y_cat
     
