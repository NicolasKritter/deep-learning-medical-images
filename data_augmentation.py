# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:05:51 2019

@author: nicolas
"""

import tensorflow as tf
import numpy as np
from prepare_data import IMG_WIDTH,IMG_HEIGHT
    
    
def flip_images(images,masks):
    uniform_random = tf.random_uniform([], 0, 1.0,seed=None)
    flip_cond = tf.less(uniform_random, .5)
    images = tf.cond(flip_cond, lambda: tf.image.flip_up_down(images), lambda: images)
    masks = tf.cond(flip_cond, lambda: tf.image.flip_up_down(masks), lambda: masks)
    uniform_random = tf.random_uniform([], 0, 1.0,seed=None)
    flip_cond = tf.less(uniform_random, .5)
    images = tf.cond(flip_cond, lambda: tf.image.flip_left_right(images), lambda: images)
    masks = tf.cond(flip_cond, lambda: tf.image.flip_left_right(masks), lambda: masks)
    return images,masks
    
def rotates_images(images, masks):
    k = tf.random_uniform([], 0, 3,dtype=tf.int32,seed=None)#random.randint(0, 3)
    images = tf.image.rot90(images,k=k,name=None)
    masks = tf.image.rot90(masks,k=k,name=None)
    return images, masks


    
def scale_imagesNMasks(images,masks):
    
    size =6#Train Size
    boxes = np.zeros((size, 4), dtype = np.float32)
    ind = np.zeros(size,dtype = np.int32)
    scale_list = [0.90, 0.75, 0.60]
    for i,scale in enumerate(scale_list):
        x1 = y1 =0.5 - 0.5 * scale
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[i] = np.array([y1, x1, y2, x2], dtype = np.float32)
        ind[i] = i
    #scale_list = tf.random.shuffle(scale_list)

    crop_size = np.array([IMG_WIDTH, IMG_HEIGHT], dtype = np.int32)
    images = tf.image.crop_and_resize(images, boxes, ind, crop_size)
    masks = tf.image.crop_and_resize(masks, boxes, ind, crop_size)
    
    return images,masks
    
def augment_imagesNMasks(images,masks):
    images,masks = rotates_images(images, masks)
    images,masks = flip_images(images, masks)
    uniform_random = tf.random_uniform([], 0, 1.0,seed=None)
    isZoom = tf.less(uniform_random, .5)
    images,masks = tf.cond(isZoom,lambda:scale_imagesNMasks(images,masks),lambda:(images,masks))
    return images,masks