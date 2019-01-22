# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 18:06:28 2018

@author: nicolas
"""
#https://towardsdatascience.com/google-deepmind-deep-learning-for-medical-image-segmentation-with-interactive-code-4634b6fd6a3a


from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import random
import matplotlib.pyplot as plt
#from prepare_data import getTrainBatch
import deep_utils as utils
import TestPrediction
from data_augmentation import augment_imagesNMasks
PICKLE_FILE = 'data.pickle'
SAVE_PATH = 'Model/unet-resnet/model.ckpt'

SEED = 42

tf.set_random_seed(SEED)

NUM_LABELS =2 #md vs not md

with open (PICKLE_FILE,'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_images']
    train_labels=save['train_mask']
    test_dataset = save['test_image']
    test_labels = save['test_labels']
    IMG_WIDTH = save['IMG_WIDTH']
    IMG_HEIGHT = save['IMG_HEIGHT']
    IMG_CHANNELS = save['IMG_CHANNELS']
    
    TRAIN_DATASET_SIZE = save['TRAIN_DATASET_SIZE']
    TEST_DATASET_SIZE = save['TEST_DATASET_SIZE']
    del save #help gc to free memory
    del PICKLE_FILE
    print('Training set (images masks)',train_dataset.shape,train_labels.shape)
    print('Test set',test_dataset.shape)





#NUM_STEP max (32 et 128*128): 275
images = train_dataset
labels = train_labels

def shuffle():
   global images,labels
   #train_dataset,train_labels=getTrainBatch()
   p = np.random.permutation(TRAIN_DATASET_SIZE)
   images = train_dataset[p]
   labels = train_labels[p]
   


def data_augment(images,masks):
    return images,masks
graph = tf.Graph()
BASE_LEARNING_RATE = 1e-4
NB_CLASSES = 3
with graph.as_default():
  
  # Input data.
  tf_train_dataset =tf.placeholder(tf.float32, [None, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS], name='data')
  tf_train_labels = tf.placeholder(tf.float32, [None, IMG_WIDTH, IMG_HEIGHT, NB_CLASSES], name='labels')
  tf_is_augment = tf.placeholder_with_default(0, [])
  tf_train_dataset,tf_train_labels = tf.cond(tf.math.greater(tf_is_augment,0),lambda:augment_imagesNMasks(tf_train_dataset,tf_train_labels),lambda:(tf_train_dataset,tf_train_labels))
  #validation
  #tf_test_dataset = tf.constant(test_dataset)
  #tf_test_labels = tf.constant(test_labels)
  
  global_step = tf.Variable(0)
  # Variables.
  #5x5 filter depth: 32 
  START_NEURON = 32 # *4 ?
# Build model
  def model(input_layer, start_neurons, num_class_, DropoutRatio = 0.5,):
    # 101 -> 50
    conv1 = utils.conv2d_3x3(start_neurons * 1, "c1")(input_layer)
    conv1 = utils.residual_block(conv1,start_neurons * 1)
    conv1 = utils.residual_block(conv1,start_neurons * 1, True)
    pool1 = utils.max_pool(2, 2)(conv1)
    pool1 = utils.drop_out(DropoutRatio/2)(pool1)

    # 50 -> 25
    conv2 = utils.conv2d_3x3(start_neurons * 2, "c2")(pool1)
    conv2 = utils.residual_block(conv2,start_neurons * 2)
    conv2 = utils.residual_block(conv2,start_neurons * 2, True)
    pool2 = utils.max_pool(2, 2)(conv2)
    pool2 = utils.drop_out(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = utils.conv2d_3x3(start_neurons * 4,"c3")(pool2)
    conv3 = utils.residual_block(conv3,start_neurons * 4)
    conv3 = utils.residual_block(conv3,start_neurons * 4, True)
    pool3 = utils.max_pool(2, 2)(conv3)
    pool3 = utils.drop_out(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = utils.conv2d_3x3(start_neurons * 8, "c4")(pool3)
    conv4 = utils.residual_block(conv4,start_neurons * 8)
    conv4 = utils.residual_block(conv4,start_neurons * 8, True)
    pool4 = utils.max_pool(2, 2)(conv4)
    pool4 = utils.drop_out(DropoutRatio)(pool4)

    # Middle
    convm = utils.conv2d_3x3(start_neurons * 16, "c5")(pool4)
    convm = utils.residual_block(convm,start_neurons * 16)
    convm = utils.residual_block(convm,start_neurons * 16, True)
    
    # 6 -> 12
    deconv4 =utils.conv2d_transpose_2x2(start_neurons * 8, (3, 3), (2, 2), "same")(convm)
    uconv4 = utils.concatenate([deconv4, conv4])
    uconv4 = utils.drop_out(DropoutRatio)(uconv4)
    
    uconv4 = utils.conv2d_3x3(start_neurons * 8,"unconv4")(uconv4)
    uconv4 = utils.residual_block(uconv4,start_neurons * 8)
    uconv4 = utils.residual_block(uconv4,start_neurons * 8, True)
    
    # 12 -> 25
    deconv3 = utils.conv2d_transpose_2x2(start_neurons * 4, (3, 3), (2, 2), "same")(uconv4)

    uconv3 = utils.concatenate([deconv3, conv3])    
    uconv3 = utils.drop_out(DropoutRatio)(uconv3)
    
    uconv3 = utils.conv2d_3x3(start_neurons * 4, "uncov3")(uconv3)
    uconv3 = utils.residual_block(uconv3,start_neurons * 4)
    uconv3 = utils.residual_block(uconv3,start_neurons * 4, True)

    # 25 -> 50
    deconv2 = utils.conv2d_transpose_2x2(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = utils.concatenate([deconv2, conv2])
        
    uconv2 = utils.drop_out(DropoutRatio)(uconv2)
    uconv2 = utils.conv2d_3x3(start_neurons * 2,"unconv2")(uconv2)
    uconv2 = utils.residual_block(uconv2,start_neurons * 2)
    uconv2 = utils.residual_block(uconv2,start_neurons * 2, True)
    
    # 50 -> 101

    deconv1 = utils.conv2d_transpose_2x2(start_neurons * 1, (3, 3), (2, 2), "same")(uconv2)
    uconv1 = utils.concatenate([deconv1, conv1])
    
    uconv1 = utils.drop_out(DropoutRatio)(uconv1)
    uconv1 = utils.conv2d_3x3(start_neurons * 1, "unconv1")(uconv1)
    uconv1 = utils.residual_block(uconv1,start_neurons * 1)
    uconv1 = utils.residual_block(uconv1,start_neurons * 1, True)
    output_layer = utils.conv2d(num_class_, (1,1), padding="same", activation=None)(uconv1)

    
    return output_layer
  logits = model(tf_train_dataset,START_NEURON,NB_CLASSES)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=logits))
  optimizer = tf.train.AdamOptimizer(BASE_LEARNING_RATE).minimize(loss)
  
  #val_model = model(tf_test_dataset,START_NEURON,NB_CLASSES)
  #val_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_test_labels, logits=val_model))

SAVE=False
RETRAIN=False
#réduire pour éviter de prendre toute la ram

NUM_STEPS =270#00
BATCH_SIZE = 2

def trainGraph(graph):
    t_loss = []
    v_loss = []
    #v_acc = []
    iteration = 0
    print("Training graph")
    with tf.Session(graph=graph) as session:
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()

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
        #batch_data, batch_labels = data_augment(batch_data,batch_labels)
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels,tf_is_augment:1}
        loss_value, _ = session.run([loss, optimizer], feed_dict=feed_dict)
        iteration+=1
        if (step % 10 == 0):
            t_loss.append(loss_value)
            #vl = val_loss.eval()
            feed_dict = {tf_train_dataset : test_dataset, tf_train_labels : test_labels}
            vl = session.run([loss], feed_dict=feed_dict)
            #acc = metrics.accuracy_score(TestPrediction.reformatForTest(test_labels).flatten(), TestPrediction.reformatForTest(res).flatten())
            v_loss.append(vl[0])
            #v_acc.append(acc)
            print(str(step) +"/"+str(NUM_STEPS)+ " training loss:", loss_value,"val_loss",vl)
        if(step % 10==0 and SAVE):
            saver.save(session,SAVE_PATH)#write_meta_graph=False
      if (SAVE and step % 10!=0) :
        saver.save(session,SAVE_PATH)
    x = np.arange(len(t_loss))*BATCH_SIZE
    plt.plot(x,t_loss)
    plt.plot(x,v_loss)
    #plt.plot(x,v_acc)
    plt.legend(['train_loss','acc'], loc='upper left')
    #plt.show()
    plt.savefig("fig.png")
    print('Done')

def testGraphOnTestSet(graph,path,test_labels,test_images):
    ix = random.randint(0, TEST_DATASET_SIZE-1)
    print(ix)
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
        
trainGraph(graph)
testGraphOnTestSet(graph,SAVE_PATH,test_labels,test_dataset)

