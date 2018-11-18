# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 18:54:53 2018

@author: nicolas
"""
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range


SEED = 42
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
tf.set_random_seed(SEED)

VGG_PATH ="./Pre-trained/VGG"
PICKLE_FILE = 'data.pickle'
#https://medium.com/nanonets/how-to-do-image-segmentation-using-deep-learning-c673cc5862ef
SAVE_PATH = 'Model/VGG/model.ckpt'

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
    print('Training set (images masks)',train_dataset.shape,train_labels.shape)
    print('Test set',test_dataset.shape)
    
    
NUMBER_OF_CLASSES = 2
IMAGE_SHAPE = (IMG_WIDTH, IMG_HEIGHT)
EPOCHS = 40
BATCH_SIZE = 2
DROPOUT = 0.75

correct_label = tf.placeholder(tf.float32, [None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], NUMBER_OF_CLASSES])
learning_rate = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)


def load_vgg(sess, VGG_PATH):
  
  # load the model and weights
  model = tf.saved_model.loader.load(sess, ['vgg16'], VGG_PATH)

  # Get Tensors to be returned from graph
  graph = tf.get_default_graph()
  image_input = graph.get_tensor_by_name('image_input:0')
  image_input = tf.reshape(image_input,[2,128,128,1])
  keep_prob = graph.get_tensor_by_name('keep_prob:0')
  layer3 = graph.get_tensor_by_name('layer3_out:0')
  layer4 = graph.get_tensor_by_name('layer4_out:0')
  layer7 = graph.get_tensor_by_name('layer7_out:0')

  return image_input, keep_prob, layer3, layer4, layer7

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
   
    # Use a shorter variable name for simplicity
    layer3, layer4, layer7 = vgg_layer3_out, vgg_layer4_out, vgg_layer7_out

    # Apply 1x1 convolution in place of fully connected layer
    fcn8 = tf.layers.conv2d(layer7, filters=num_classes, kernel_size=1, name="fcn8")

    # Upsample fcn8 with size depth=(4096?) to match size of layer 4 so that we can add skip connection with 4th layer
    fcn9 = tf.layers.conv2d_transpose(fcn8, filters=layer4.get_shape().as_list()[-1],
    kernel_size=4, strides=(2, 2), padding='SAME', name="fcn9")

    # Add a skip connection between current final layer fcn8 and 4th layer
    fcn9_skip_connected = tf.add(fcn9, layer4, name="fcn9_plus_vgg_layer4")

    # Upsample again
    fcn10 = tf.layers.conv2d_transpose(fcn9_skip_connected, filters=layer3.get_shape().as_list()[-1],
    kernel_size=4, strides=(2, 2), padding='SAME', name="fcn10_conv2d")

    # Add skip connection
    fcn10_skip_connected = tf.add(fcn10, layer3, name="fcn10_plus_vgg_layer3")

    # Upsample again
    fcn11 = tf.layers.conv2d_transpose(fcn10_skip_connected, filters=num_classes,
    kernel_size=16, strides=(8, 8), padding='SAME', name="fcn11")

    return fcn11

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
  
  # Reshape 4D tensors to 2D, each row represents a pixel, each column a class
  logits = tf.reshape(nn_last_layer, (-1, num_classes), name="fcn_logits")
  correct_label_reshaped = tf.reshape(correct_label, (-1, num_classes))

  # Calculate distance from actual labels using cross entropy
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=correct_label_reshaped[:])
  # Take mean for total loss
  loss_op = tf.reduce_mean(cross_entropy, name="fcn_loss")

  # The model implements this operation to find the weights/parameters that would yield correct pixel labels
  train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name="fcn_train_op")

  return logits, train_op, loss_op

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
             cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):

  keep_prob_value = 0.5
  learning_rate_value = 0.001
  for epoch in range(epochs):
      # Create function to get batches
      total_loss = 0
      for X_batch, gt_batch in get_batches_fn(batch_size):

          loss, _ = sess.run([cross_entropy_loss, train_op],
          feed_dict={input_image: X_batch, correct_label: gt_batch,
          keep_prob: keep_prob_value, learning_rate:learning_rate_value})

          total_loss += loss;

      print("EPOCH {} ...".format(epoch + 1))
      print("Loss = {:.3f}".format(total_loss))
      print()

def shuffle():
   global images,labels
   p = np.random.permutation(len(train_dataset))
   images = train_dataset[p]
   labels = train_labels[p]

NUM_STEPS = 10
SAVE=True
def get_batches_fn(batch_size):
    iteration = 0
    resdata = []
    reslabel = []
    for step in range(NUM_STEPS):
        if iteration>67:
            shuffle()
            iteration = 0
        offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)
        batch_data = train_dataset[offset:(offset + BATCH_SIZE), :]
        batch_labels = train_labels[offset:(offset + BATCH_SIZE), :]
        iteration +=1
        resdata.append(batch_data)
        reslabel.append(batch_labels)


def run():
  iteration = 0
  with tf.Session() as session:
        
    # Returns the three layers, keep probability and input layer from the vgg architecture
    image_input, keep_prob, layer3, layer4, layer7 = load_vgg(session, VGG_PATH)

    # The resulting network architecture from adding a decoder on top of the given vgg model
    model_output = layers(layer3, layer4, layer7, NUMBER_OF_CLASSES)

    # Returns the output logits, training operation and cost operation to be used
    # - logits: each row represents a pixel, each column a class
    # - train_op: function used to get the right parameters to the model to correctly label the pixels
    # - cross_entropy_loss: function outputting the cost which we are minimizing, lower cost should yield higher accuracy
    logits, train_op, cross_entropy_loss = optimize(model_output, correct_label, learning_rate, NUMBER_OF_CLASSES)
    
    # Initialize all variables
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())

    print("Model build successful, starting training")

    # Train the neural network
    """train_nn(session, EPOCHS, BATCH_SIZE, get_batches_fn, 
             train_op, cross_entropy_loss, image_input,
             correct_label, keep_prob, learning_rate)
"""
    saver = tf.train.Saver()
    if RETRAIN:
         saver.restore(session, SAVE_PATH)
      #Epoch
    keep_prob_value = 0.5
    learning_rate_value = 0.001
    for step in range(NUM_STEPS):
        if iteration>TRAIN_DATASET_SIZE:
            shuffle()
            iteration = 0
        offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)
        batch_data = train_dataset[offset:(offset + BATCH_SIZE), :]
        batch_labels = train_labels[offset:(offset + BATCH_SIZE), :]
        loss_value, _ = session.run([cross_entropy_loss, train_op],
          feed_dict={image_input: batch_data, correct_label: batch_labels,
          keep_prob: keep_prob_value, learning_rate:learning_rate_value})
        iteration+=1
        if (step % 10 == 0):
            print(str(step) +"/"+str(NUM_STEPS)+ " training loss:", str(loss_value))
    #saves a model every 2 hours and maximum 4 latest models are saved.
        if(step % 10==0):
            saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=1)#write_meta_graph=False
    if SAVE:
          saver.save(session,SAVE_PATH)
    print('Done')

RETRAIN = False
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
      keep_prob_value = 0.5
      learning_rate_value = 0.001
      for step in range(NUM_STEPS):
        if iteration>TRAIN_DATASET_SIZE:
            shuffle()
            iteration = 0
        offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)
        batch_data = train_dataset[offset:(offset + BATCH_SIZE), :]
        batch_labels = train_labels[offset:(offset + BATCH_SIZE), :]
        loss_value, _ = session.run([cross_entropy_loss, train_op],
          feed_dict={input_image: batch_data, correct_label: batch_labels,
          keep_prob: keep_prob_value, learning_rate:learning_rate_value})
        iteration+=1
        if (step % 10 == 0):
            print(str(step) +"/"+str(NUM_STEPS)+ " training loss:", str(loss_value))
#saves a model every 2 hours and maximum 4 latest models are saved.
        if(step % 10==0):
            saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=1)#write_meta_graph=False
      if SAVE:
          saver.save(session,SAVE_PATH)
      print('Done')
run()