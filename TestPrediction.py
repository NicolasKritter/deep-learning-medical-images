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
    imshow(img)
    plt.subplot(222)
    plt.title("true mask")
    imshow(true_mask)
    plt.subplot(223)
    plt.title("produced mask")
    imshow(check_train_mask)
    plt.show()
     
    return true_mask,check_train_mask

def reformatForTest(t):
    return np.argmax(t,axis=-1)

def toLabels(pred,treshold):
    np.putmask(pred,pred<treshold,1)
    np.putmask(pred,pred>=treshold,0)
    return pred
    


def testGraphOnTestSet(graph,path,test_labels,test_images,TEST_DATASET_SIZE):
    ix = random.randint(0, TEST_DATASET_SIZE-1)
    img = test_images[ix];
    check_data = np.expand_dims(np.array(img), axis=0)
    print(test_labels.shape,test_labels.min(),test_labels.max())
    with tf.Session(graph=graph) as session:

        saver = tf.train.Saver()
        saver.restore(session, path)
        check_train = {tf_train_dataset:check_data}
        check_mask = session.run(logits,feed_dict=check_train)
        true_mask = test_labels[ix]
        true_mask = reformatForTest(true_mask)
        check_mask = reformatForTest(check_mask[0])
        plotImgvsTMaskVsPredMask(img,true_mask,check_mask)
        fscore, acc, recall, pres, cmat = evaluate(true_mask.flatten(),check_mask.flatten() ,2)
        print('F-score: '+str(fscore)+'\tacc: '+str(acc),'\trecall: '+str(recall),'\tprecision: '+str(pres))
        print(cmat)