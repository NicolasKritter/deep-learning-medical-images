# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 15:03:50 2019

@author: nicolas
"""

import tensorflow as tf
from tensorflow.python.ops import array_ops

def deconv2d(input_tensor, filter_size, output_size, out_channels, in_channels, name, strides = [1, 1, 1, 1]):
    dyn_input_shape = tf.shape(input_tensor)
    batch_size = dyn_input_shape[0]
    out_shape = tf.stack([batch_size, output_size, output_size, out_channels])
    filter_shape = [filter_size, filter_size, out_channels, in_channels]
    w = tf.get_variable(name=name, shape=filter_shape)
    h1 = tf.nn.conv2d_transpose(input_tensor, w, out_shape, strides, padding='same')
    return h1
    
def conv2d( depth, kernel=(1,1), strides=(1, 1),activation=tf.nn.relu, padding="same",name="conv2d"):
    return tf.layers.Conv2D(filters=depth, kernel_size=kernel, strides=strides, padding=padding, activation=activation, name=name)

def conv2d_3x3(filters, name="name"):
    return tf.layers.Conv2D(filters=filters, kernel_size=(3,3), activation=tf.nn.relu, padding='same', name=name)

def max_pool(x,y):
    return tf.layers.MaxPooling2D((x,y), strides=2, padding='same') 

def drop_out(rate):
    return tf.layers.Dropout(rate)

def conv2d_transpose_2x2(filters, kernel_size=(3, 3), strides=(2, 2), padding="same",name="conv2dT"):
    return tf.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, name=name)

def concatenate(branches):
    return array_ops.concat(branches, 3)
    
def BatchActivate(x):
    x = tf.nn.relu(x)
    x = tf.layers.batch_normalization(x)
#    x = keras.layers.Activation('relu')(x)
    return x

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = conv2d(filters, size, strides=strides, padding=padding)(x)
    if activation == True:
        x = BatchActivate(x)
    return x

def residual_block(blockInput, num_filters=16, batch_activate = False):
    x = BatchActivate(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = tf.math.add(x,blockInput)
    if batch_activate:
        x = BatchActivate(x)
    return x