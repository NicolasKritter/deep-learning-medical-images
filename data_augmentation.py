# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:05:51 2019

@author: nicolas
"""

import tensorflow as tf


    
    
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

def random_crop_and_pad_image_and_labels(image, labels, size):

  combined = tf.concat([image, labels], axis=2)
  image_shape = tf.shape(image)
  combined_pad = tf.image.pad_to_bounding_box(
      combined, 0, 0,
      tf.maximum(size[0], image_shape[0]),
      tf.maximum(size[1], image_shape[1]))
  last_label_dim = tf.shape(labels)[-1]
  last_image_dim = tf.shape(image)[-1]
  combined_crop = tf.random_crop(combined_pad,size=tf.concat([size, [last_label_dim + last_image_dim]],axis=0))
  return (combined_crop[:, :, :last_image_dim],
          combined_crop[:, :, last_image_dim:])

def scale_imagesNMasks(images,masks):
    """original_size = [IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS]
    seed =  np.random.randint(1234)
    scale_factor  = tf.random_uniform([], 0.6, 0.91,seed=None)
    crop_size = [tf.to_int32(IMG_HEIGHT*scale_factor), tf.to_int32(IMG_WIDTH*scale_factor)]
    """
    """images = tf.random_crop(images, size = crop_size, seed = seed)
    masks  =tf.random_crop(masks, size = crop_size, seed = seed)"""
    """    
    images,masks = random_crop_and_pad_image_and_labels(images,masks,crop_size)
    images = tf.images.resize_images(images, size = original_size)
    masks = tf.images.resize_images(masks, size = original_size)"""
    """
    size =6#Train Size
    scale_list = [0.90, 0.75, 0.60]
    k = tf.random_uniform([], 0,len(scale_list)-1,dtype=tf.int32,seed=None)
    boxes = np.zeros((size, 4), dtype = np.float32)
    #scale_list = tf.random.shuffle(scale_list)
    scale = scale_list[tf.to_int64(k)]
    x1 = y1 = 0.5 - 0.5 * scale
    x2 = y2 = 0.5 + 0.5 * scale
    boxes[0] = np.array([y1, x1, y2, x2], dtype = np.float32)
    crop_size = np.array([IMG_WIDTH, IMG_HEIGHT], dtype = np.int32)
    images = tf.image.crop_and_resize(images, boxes, np.zeros(size,dtype = np.int32), crop_size)
    masks = tf.image.crop_and_resize(masks, boxes, np.zeros(size, dtype = np.int32), crop_size)"""
    return images,masks
    
def augment_imagesNMasks(images,masks):
    images,masks = rotates_images(images, masks)
    images,masks = flip_images(images, masks)
    uniform_random = tf.random_uniform([], 0, 1.0,seed=None)
    isZoom = tf.less(uniform_random, .7)
    images,masks = tf.cond(isZoom,lambda:scale_imagesNMasks(images,masks),lambda:(images,masks))
    return images,masks