# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 18:30:02 2018

@author: nicolas
"""

import tensorflow as tf
import sklearn.metrics as metrics
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import  imshow
import random
from six.moves import cPickle as pickle

def evaluate(actual,pred,num_class):
    fscore = metrics.f1_score(actual, pred, average='macro')
    acc = metrics.accuracy_score(actual, pred)
    pres = metrics.precision_score(actual, pred,average='macro')
    recall = metrics.recall_score(actual, pred, average='macro')
    
    labl=[]
    
    for i in range(num_class):
        labl.append(i)
   
    cm = metrics.confusion_matrix(actual,pred,labels=labl)
    fscoref=round(fscore,3)
    accf=round(acc,3)
    recallf=round(recall,3)
    presf=round(pres,3)
    
    print (classification_report(actual, pred))
    return fscoref, accf, recallf,presf, cm
    
          
def plotImgvsTMaskVsPredMask(img,true_mask,check_train_mask):
    plt.figure(1)
    print("original image")
    plt.subplot(221)
    plt.title("original image")
    imshow(img.astype(float)[:, :, 0])
    plt.subplot(222)
    plt.title("true mask")
    imshow(true_mask)
    plt.subplot(223)
    plt.title("produced mask")

    imshow(check_train_mask)
    plt.show()
     
    return true_mask,check_train_mask

def reformatForTest(t):
    return t.squeeze().astype(np.uint8)

def toLabels(pred,treshold):
    np.putmask(pred,pred<treshold,1)
    np.putmask(pred,pred>=treshold,0)
    return pred
    
