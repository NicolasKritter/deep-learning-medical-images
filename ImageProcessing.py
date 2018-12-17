# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 17:37:17 2018

@author: nicolas
"""
import numpy as np

from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage.io import  imshow
import cv2

#OpenCV: BGR
LOWER_RED = np.array([27,0,130])#136,0,27
UPPER_RED = np.array([29,1,136])

LOWER_YELLOW = np.array([0,254,254])
UPPER_YELLOW = np.array([0,255,255])

#PATH
TEST_PATH= 'test/1.tif'

#hot encoding
HOT_COEUR = np.array([0,1,0])#1 #yellow
HOT_GAINE = np.array([1,0,0])

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
        m1 = cv2.bitwise_or(m1,masks[i])
    return m1
    
def getRedMask(img):
    red_mask = getCleanMask(img,LOWER_RED,UPPER_RED)
    red_mask[np.where((red_mask >= LOWER_RED).all(axis = 2))] = HOT_GAINE
    return  red_mask
def getYellowMask(img):
    yellow_mask = getCleanMask(img,LOWER_YELLOW,UPPER_YELLOW)
    yellow_mask[np.where((yellow_mask >= LOWER_YELLOW).all(axis = 2))] = HOT_COEUR
    return yellow_mask   

def getFullMask(path,toOne=True):
    if not toOne:
        global HOT_COEUR,HOT_GAINE
        HOT_COEUR*=255
        HOT_GAINE*=255
    img =  cv2.imread(path)[:,:,:3]
    m1 = getRedMask(img)
    m2 = getYellowMask(img)
    return mergeMasks([m1,m2])
def test():
    img = getFullMask(TEST_PATH,False)
    plt.title("testing ")
    imshow(img)
    plt.show()

    
test()

