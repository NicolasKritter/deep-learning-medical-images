# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 15:57:51 2018

@author: nicolas
"""

import numpy as np

import math
import random
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.python.ops import array_ops
from six.moves import cPickle as pickle
import TestPrediction
from ImageProcessing import rotation

PICKLE_FILE = 'data.pickle'
SAVE_PATH = 'Model/unet/model.ckpt'

with open (PICKLE_FILE,'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_images']
    train_labels=save['train_mask']
    test_dataset = save['test_image']
    test_labels = save['test_labels']
    IMG_WIDTH = save['IMG_WIDTH']
    IMG_HEIGHT = save['IMG_WIDTH']
    IMG_CHANNELS = save['IMG_CHANNELS']
    
    TRAIN_DATASET_SIZE = save['TRAIN_DATASET_SIZE']
    TEST_DATASET_SIZE = save['TEST_DATASET_SIZE']
    del save #help gc to free memory
    del PICKLE_FILE
    print('Training set (images, masks)',train_dataset.shape,train_labels.shape)
    print('Test set',test_dataset.shape)
    
SEED = 42
tf.set_random_seed(SEED)

def deconv2d(input_tensor, filter_size, output_size, out_channels, in_channels, name, strides = [1, 1, 1, 1]):
    dyn_input_shape = tf.shape(input_tensor)
    batch_size = dyn_input_shape[0]
    out_shape = tf.stack([batch_size, output_size, output_size, out_channels])
    filter_shape = [filter_size, filter_size, out_channels, in_channels]
    w = tf.get_variable(name=name, shape=filter_shape)
    h1 = tf.nn.conv2d_transpose(input_tensor, w, out_shape, strides, padding='SAME')
    return h1

def conv2d( depth, kernel=(1,1), strides=(1, 1),activation=tf.nn.relu, padding="same",name="conv2d"):
    return tf.layers.Conv2D(filters=depth, kernel_size=kernel, strides=strides, padding=padding, activation=activation, name=name)

def conv2d_3x3(depth, name):
    return tf.layers.Conv2D(filters=depth, kernel_size=(3,3), activation=tf.nn.relu, padding='same', name=name)

def max_pool():
    return tf.layers.MaxPooling2D((2,2), strides=2, padding='same') 

def drop_out(rate):
    return tf.layers.Dropout(rate)

def conv2d_transpose_2x2(filters, name):
    return tf.layers.Conv2DTranspose(filters=filters, kernel_size=(2, 2), strides=(2, 2), padding='same', name=name)

def concatenate(branches):
    return array_ops.concat(branches, 3)


NUM_STEPS = 100#00

images = train_dataset
labels = train_labels

def shuffle():
   global images,labels
   p = np.random.permutation(len(train_dataset))
   images = train_dataset[p]
   labels = train_labels[p]

SAVE=True
RETRAIN=False
#réduire pour éviter de prendre toute la ram

BATCH_SIZE = 2
def trainGraph(graph):
    t_loss = []
    v_loss = []
    iteration = 0
    print("Training graph")
    with tf.Session(graph=graph) as session:
      tf.global_variables_initializer().run()
      print('Initialized')
      saver = tf.train.Saver()
      #saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=1)

      if RETRAIN:
         saver.restore(session, SAVE_PATH)
      #Epoch
      for step in range(NUM_STEPS):
        if iteration>TRAIN_DATASET_SIZE:
            shuffle()
            iteration = 0
        offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)
        batch_data = train_dataset[offset:(offset + BATCH_SIZE), :]
        batch_labels = train_labels[offset:(offset + BATCH_SIZE), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        loss_value, _ = session.run([loss, optimizer], feed_dict=feed_dict)
        iteration+=1
        if (step % 10 == 0):
            t_loss.append(loss_value)
            vl = val_loss.eval()
            v_loss.append(vl)
            print(str(step) +"/"+str(NUM_STEPS)+ " training loss:", loss_value,"val_loss",vl)
#saves a model every 2 hours and maximum 4 latest models are saved.
        if(step % 10==0 and SAVE):
            saver.save(session,SAVE_PATH)#write_meta_graph=False
      if (SAVE and step % 10!=0) :
        saver.save(session,SAVE_PATH)
    x = np.arange(len(v_loss))*BATCH_SIZE
    plt.plot(x,t_loss)
    plt.plot(x,v_loss)
    plt.legend(['train_loss','val_loss'], loc='upper left')
    plt.show()
    print('Done')

def testGraphOnTestSet(graph,path,test_labels,test_images):
    ix = random.randint(0, TEST_DATASET_SIZE-1)
    img = test_images[ix];
    check_data = np.expand_dims(np.array(img), axis=0)
    with tf.Session(graph=graph) as session:

        saver = tf.train.Saver()
        saver.restore(session, path)
        check_train = {tf_train_dataset:check_data}
        check_mask = session.run(logits,feed_dict=check_train)
        true_mask = test_labels[ix]
        true_mask = TestPrediction.reformatForTest(true_mask)
        check_mask = TestPrediction.reformatForTest(check_mask[0])
        TestPrediction.plotImgvsTMaskVsPredMask(img,true_mask,check_mask)
        fscore, acc, recall, pres, cmat = TestPrediction.evaluate(true_mask.flatten(),check_mask.flatten() ,NB_CLASSES)
        print('F-score: '+str(fscore)+'\tacc: '+str(acc),'\trecall: '+str(recall),'\tprecision: '+str(pres))
        print(cmat)

graph = tf.Graph()
BASE_LEARNING_RATE = 1e-4
NB_CLASSES = 3
with graph.as_default():

  # Input data.
  tf_train_dataset =tf.placeholder(tf.float32, [None, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS], name='data')
  tf_train_labels = tf.placeholder(tf.uint8, [None, IMG_WIDTH, IMG_HEIGHT, NB_CLASSES], name='labels')
  tf_test_dataset = tf.constant(test_dataset)
  tf_test_labels = tf.constant(test_labels)
  class_weights = tf.constant([0.5,0.8,1])
#TODO weihts
  #tf.multiply(tf_train_labels, class_weights)
  global_step = tf.Variable(0)
  # Variables.
  #5x5 filter depth: 32 
  
  def model(input_layer,num_class, train=False):
    c1 = conv2d_3x3(32, "c1") (input_layer)
    c1 = drop_out(0.1) (c1)
    c1 = conv2d_3x3(32, "c1") (c1)
    p1 = max_pool() (c1)

    c2 = conv2d_3x3(64, "c2") (p1)
    c2 = drop_out(0.1) (c2)
    c2 = conv2d_3x3(64, "c2") (c2)
    p2 = max_pool() (c2)

    c3 = conv2d_3x3(128, "c3") (p2)
    c3 = drop_out(0.1) (c3)
    c3 = conv2d_3x3(128, "c3") (c3)
    p3 = max_pool() (c3)

    c4 = conv2d_3x3(256, "c4") (p3)
    c4 = drop_out(0.2) (c4)
    c4 = conv2d_3x3(256, "c4") (c4)
    p4 = max_pool() (c4)
    
    c5 = conv2d_3x3(512, "c5") (p4)
    c5 = drop_out(0.1) (c5)
    c5 = conv2d_3x3(512, "c5") (c5)

    u6 = conv2d_transpose_2x2(256, "u6") (c5)
    u6 = concatenate([u6, c4])
    c6 = conv2d_3x3(256, "c6") (u6)
    c6 = drop_out(0.2) (c6)
    c6 = conv2d_3x3(256, "c6") (c6)

    u7 = conv2d_transpose_2x2(128, "u7") (c6)
    u7 = concatenate([u7, c3])
    c7 = conv2d_3x3(128, "c7") (u7)
    c7 = drop_out(0.1) (c7)
    c7 = conv2d_3x3(128, "c7") (c7)

    u8 = conv2d_transpose_2x2(64, "u8") (c7)
    u8 = concatenate([u8, c2])
    c8 = conv2d_3x3(64, "c8") (u8)
    c8 = drop_out(0.1) (c8)
    c8 = conv2d_3x3(64, "c8") (c8)

    u9 = conv2d_transpose_2x2(32, "u9") (c8)
    u9 = concatenate([u9, c1])
    c9 = conv2d_3x3(32, "c9") (u9)
    c9 = drop_out(0.1) (c9)
    c9 = conv2d_3x3(32, "c9") (c9)
    if train:
      c9 = tf.nn.dropout(c9, 0.5, seed=SEED)
    
    output_layer = conv2d(num_class, kernel=(1,1), padding="same", activation=None)(c9)
    return output_layer
  logits = model(tf_train_dataset,NB_CLASSES,True)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=logits))
  optimizer =  tf.train.AdamOptimizer(BASE_LEARNING_RATE).minimize(loss)
  val_loss =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_test_labels, logits=model(tf_test_dataset,NB_CLASSES,True)))


      


trainGraph(graph)
testGraphOnTestSet(graph,SAVE_PATH,test_labels,test_dataset)
#TestPrediction.testGraphOnTestSet(graph,SAVE_PATH,test_labels,test_dataset,TEST_DATASET_SIZE)

