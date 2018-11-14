# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 15:57:51 2018

@author: nicolas
"""

import numpy as np

import math
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from skimage.io import  imshow

from tensorflow.python.ops import array_ops
from six.moves import cPickle as pickle

PICKLE_FILE = 'data.pickle'
SAVE_PATH = 'Model/model.ckpt'

with open (PICKLE_FILE,'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_images']
    train_labels=save['train_mask']
    test_dataset = save['test_image']
    
    IMG_WIDTH = save['IMG_WIDTH']
    IMG_HEIGHT = save['IMG_WIDTH']
    IMG_CHANNELS = save['IMG_CHANNELS']
    
    TRAIN_DATASET_SIZE = save['TRAIN_DATASET_SIZE']
    TEST_DATASET_SIZE = save['TEST_DATASET_SIZE']
    del save #help gc to free memory
    del PICKLE_FILE
    print('Training set (images masks)',train_dataset.shape,train_labels.shape)
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

def conv2d(input_tensor, depth, kernel, name, strides=(1, 1), padding="SAME"):
    return tf.layers.conv2d(input_tensor, filters=depth, kernel_size=kernel, strides=strides, padding=padding, activation=tf.nn.relu, name=name)

def conv2d_3x3(filters, name):
    return tf.layers.Conv2D(filters=filters, kernel_size=(3,3), activation=tf.nn.relu, padding='same', name=name)

def max_pool():
    return tf.layers.MaxPooling2D((2,2), strides=2, padding='same') 

def drop_out(rate):
    return tf.layers.Dropout(rate)

def conv2d_transpose_2x2(filters, name):
    return tf.layers.Conv2DTranspose(filters=filters, kernel_size=(2, 2), strides=(2, 2), padding='same', name=name)

def concatenate(branches):
    return array_ops.concat(branches, 3)
    
def sigmoid(x):
    x = x/10
    return 1 / (1 + math.exp(-x))



NUM_STEPS = 100#00
images = train_dataset
labels = train_labels

def shuffle():
   global images,labels
   p = np.random.permutation(len(train_dataset))
   images = train_dataset[p]
   labels = train_labels[p]

SAVE=True
RETRAIN=True
#réduire pour éviter de prendre toute la ram
BATCH_SIZE = 2

def trainGraph(graph):
    iteration = 0
    print("Training graph")
    with tf.Session(graph=graph) as session:
      tf.global_variables_initializer().run()
      saver = tf.train.Saver()
      print('Initialized')
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
        if (step % 50 == 0):
            print(str(step) +"/"+str(NUM_STEPS)+ " training loss:", str(loss_value))
#saves a model every 2 hours and maximum 4 latest models are saved.
        if(step % 10==0):
            saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=1)#write_meta_graph=False
      if SAVE:
          saver.save(session,SAVE_PATH)
      print('Done')

def testGraphOnTestSet(graph):
    ix = random.randint(0, TEST_DATASET_SIZE-1)
    check_data = np.expand_dims(np.array(images[ix]), axis=0)
    with tf.Session(graph=graph) as session:
        saver = tf.train.Saver()
        saver.restore(session, SAVE_PATH)
        check_train = {tf_train_dataset:check_data}
        check_train_mask = session.run(logits,feed_dict=check_train)
        true_mask = labels[ix].astype(float)[:, :, 0]
        print("original image")
        imshow(images[ix].astype(float)[:, :, 0])
        plt.show()
        print("true mask")
        print(true_mask.shape)
        imshow(true_mask.squeeze().astype(np.uint8))
        plt.show()
        print("produced mask")
        print(check_train_mask.shape)
        imshow(check_train_mask.squeeze().astype(np.uint8))
        plt.show()

"""
def testGraph(graph):
    ix = random.randint(0, TEST_DATASET_SIZE-1)
    test_image = test_dataset[ix].astype(float)[:, :, 0]
    imshow(test_image)
    plt.show()
    with tf.Session(graph=graph) as session:
        saver = tf.train.Saver()
        saver.restore(session, SAVE_PATH)
        test_image = np.reshape(test_image, [-1, IMG_WIDTH , IMG_HEIGHT, 1])
        test_data = {tf_train_dataset : test_image}
        test_mask = session.run([logits],feed_dict=test_data)
        test_mask = np.reshape(np.squeeze(test_mask), [IMG_WIDTH , IMG_HEIGHT, 1])
        for i in range(IMG_WIDTH):
            for j in range(IMG_HEIGHT):
                    test_mask[i][j] = int(sigmoid(test_mask[i][j])*255)
        imshow(test_mask.squeeze().astype(np.uint8))
        plt.show()
"""

graph = tf.Graph()
BASE_LEARNING_RATE = 1e-4

with graph.as_default():

  # Input data.
  tf_train_dataset =tf.placeholder(tf.float32, [None, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS], name='data')
  tf_train_labels = tf.placeholder(tf.float32, [None, IMG_WIDTH, IMG_HEIGHT, 1], name='labels')
  """tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)"""
  global_step = tf.Variable(0)
  # Variables.
  #5x5 filter depth: 32 
  
  def model(data, train=False):
    c1 = conv2d_3x3(32, "c1") (data)
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
    return tf.layers.Conv2D(1,(1,1))(c9)

  # Training computation: logits + cross-entropy loss.
  logits = model(tf_train_dataset, True)
  loss = tf.losses.sigmoid_cross_entropy(tf_train_labels, logits)

  # Regularization 

  # Optimizer.

  #learning_rate = tf.train.exponential_decay(BASE_LEARNING_RATE,global_step, DECAY_STEP, DECAY_RATE,staircase=True)
      
  #optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss,global_step=global_step) 
  optimizer = tf.train.AdamOptimizer(BASE_LEARNING_RATE).minimize(loss)
 # Predictions for the training, validation, and test data.
  """train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model(tf_test_dataset))"""


      

#testGraph(graph)
#trainGraph(graph)
testGraphOnTestSet(graph)
