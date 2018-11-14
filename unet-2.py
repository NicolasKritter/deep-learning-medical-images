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
from sklearn.utils import shuffle

import matplotlib.pyplot as plt

PICKLE_FILE = 'data.pickle'
SAVE_PATH = 'Model/unet-2/model.ckpt'

SEED = 42

IMG_CHANNELS = 1
tf.set_random_seed(SEED)

NUM_LABELS =2 #md vs not md

with open (PICKLE_FILE,'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_images']
    train_labels=save['train_mask']
    test_dataset = save['test_image']
    IMG_WIDTH = save['IMG_WIDTH']
    IMG_HEIGHT = save['IMG_WIDTH']
    del save #help gc to free memory
    del PICKLE_FILE
    print('Training set (images masks)',train_dataset.shape,train_labels.shape)
    print('Test set',test_dataset.shape)
    
    
    
BATCH_SIZE = 32

NUM_STEPS = 10#000
images = train_dataset
labels = train_labels


SAVE=True
def trainGraph(graph):
    iteration = 0
    print("Training graph")
    with tf.Session(graph=graph) as session:
      tf.global_variables_initializer().run()
      print('Initialized')
      #Epoch
      for step in range(NUM_STEPS):
        if iteration>67:
           # shuffle()
            iteration = 0
        offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)
        batch_data = train_dataset[offset:(offset + BATCH_SIZE), :]
        batch_labels = train_labels[offset:(offset + BATCH_SIZE), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        loss_value, _ = session.run([loss, optimizer], feed_dict=feed_dict)
        iteration+=1
        if (step % 500 == 0):
            print(str(step) +"/"+str(NUM_STEPS)+ " training loss:", str(loss_value))
      if SAVE:
          save_path =  saver.save(session,SAVE_PATH)
          print(save_path)
      print('Done')



def tf_relu(x): return tf.nn.relu(x)
def d_tf_relu(s): return tf.cast(tf.greater(s,0),dtype=tf.float32)

def tf_tanh(x): return tf.nn.tanh(x)
def d_tf_tanh(x): return 1-tf_tanh(x) ** 2

def tf_softmax(x): return tf.nn.softmax(x)


# --- make class ---
class CNN_Layer():
    
    def __init__(self,kernel,in_c,out_c):
        self.w = tf.Variable(tf.random_normal([kernel,kernel,in_c,out_c],stddev=0.005),name="w1")

    def feedforward(self,input,stride=1,dilate=1):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides = [1,stride,stride,1],padding='SAME')#input, filter,stride,padding
        self.layerA = tf_relu(self.layer)
        return self.layerA
    
    def backprop(self,gradient):
        
        grad_part_1 = gradient 
        grad_part_2 = d_tf_relu(self.layer)
        grad_part_3 = self.input

        grad_middle = tf.multiply(grad_part_1,grad_part_2)
        grad = tf.nn.conv2d_backprop_filter(
            input = grad_part_3,
            filter_sizes = self.w.shape,
            out_backprop = grad_middle,
            strides=[1,1,1,1],padding='SAME'
        )

        grad_pass = tf.nn.conv2d_backprop_input(
            input_sizes=[batch_size] + list(self.input.shape[1:]),
            filter = self.w,
            out_backprop = grad_middle,
            strides=[1,1,1,1],padding='SAME'
        )

        update_w = []
        update_w.append(tf.assign(self.w, self.w - learning_rate * grad))
        return grad_pass,update_w

class FNN_layer():
    def __init__(self,input_dim,hidden_dim):
        self.w = tf.Variable(tf.truncated_normal([input_dim,hidden_dim], stddev=0.005),name="w2")

    def feedforward(self,input=None):
        self.input = input
        self.layer = tf.matmul(input,self.w)
        self.layerA = tf_tanh(self.layer)
        return self.layerA

    def backprop(self,gradient=None):
        grad_part_1 = gradient
        grad_part_2 = self.d_act(self.layer)
        grad_part_3 = self.input 

        grad_x_mid = tf.multiply(grad_part_1,grad_part_2)
        grad = tf.matmul(tf.transpose(grad_part_3),grad_x_mid)
        grad_pass = tf.matmul(tf.multiply(grad_part_1,grad_part_2),tf.transpose(self.w))

        update_w = []
        update_w.append(tf.assign(self.w, self.w - learning_rate * grad))
        return grad_pass,update_w
    

# --- hyper ---
num_epoch = 100
init_lr = 0.01
batch_size = 2

# --- make layer ---
l1 = CNN_Layer(5,1,10) #kernel,filter input size, filter output size 
l2 = CNN_Layer(5,10,25)

FNN_Input = IMG_WIDTH * IMG_HEIGHT * 25 
l3 = FNN_layer(FNN_Input,1000)
l4 = FNN_layer(1000,IMG_WIDTH * IMG_HEIGHT)

# ---- make graph ----
tf_train_dataset = tf.placeholder(shape=[None,IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS],dtype=tf.float32,name="tr")
tf_train_labels = tf.placeholder(shape=[None,IMG_WIDTH*IMG_HEIGHT],dtype=tf.float32,name="tl")

layer1 = l1.feedforward(tf_train_dataset) #conv2D -> relu
layer2 = l2.feedforward(layer1)#conv 2D-> reul
layer3_Input = tf.reshape(layer2,[batch_size,-1])
layer3 = l3.feedforward(layer3_Input) #mat mult
layer4 = l4.feedforward(layer3)#mattmul

final_softmax = tf_softmax(layer4)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer4,labels=tf_train_labels))
auto_train = tf.train.GradientDescentOptimizer(learning_rate=init_lr).minimize(cost)


SAVE=True
num_epoch = num_epoch/(len(train_dataset)-batch_size)
# --- start session ---
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    for iter in range(num_epoch):
        print("iteration: ",iter)
        # train
        for current_batch_index in range(0,len(train_dataset)-batch_size,batch_size):
            current_batch = train_dataset[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_label = np.reshape(train_labels[current_batch_index:current_batch_index+batch_size,:,:,:],(batch_size,-1))
            sess_results = session.run([cost,auto_train],feed_dict={tf_train_dataset:current_batch,tf_train_labels:current_label})
            print(current_batch_index,' Iter: ', iter, " Cost:  %.32f"% sess_results[0])

        print('\n-----------------------')
        train_images,train_labels = shuffle(train_dataset,train_labels)
        if(iter % 10==0):
            saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=1)#write_meta_graph=False
      
        if iter % 10 == 0:
            test_example =   train_images[:batch_size,:,:,:]
            test_example_gt = np.reshape(train_labels[:batch_size,:,:,:],(batch_size,-1))
            sess_results = session.run([final_softmax],feed_dict={tf_train_dataset:test_example})

            sess_results = np.reshape(sess_results[0][0],(IMG_WIDTH,IMG_HEIGHT))
            test_example = test_example[0,:,:,:]
            test_example_gt =np.reshape( test_example_gt[0],(IMG_WIDTH,IMG_HEIGHT))

            plt.figure()
            plt.imshow(np.squeeze(test_example),cmap='gray')
            plt.axis('off')
            plt.title('epoch_'+str(iter)+'Original Image')
            plt.savefig('train_change/epoch_'+str(iter)+"a_Original_Image.png")

            plt.figure()
            plt.imshow(np.squeeze(test_example_gt),cmap='gray')
            plt.axis('off')
            plt.title('epoch_'+str(iter)+'Ground Truth Mask')
            plt.savefig('train_change/epoch_'+str(iter)+"b_Original_Mask.png")

            plt.figure()
            plt.imshow(np.squeeze(sess_results),cmap='gray')
            plt.axis('off')
            plt.title('epoch_'+str(iter)+'Generated Mask')
            plt.savefig('train_change/epoch_'+str(iter)+"c_Generated_Mask.png")

            plt.figure()
            plt.imshow(np.multiply(np.squeeze(test_example),np.squeeze(test_example_gt)),cmap='gray')
            plt.axis('off')
            plt.title('epoch_'+str(iter)+"Ground Truth Overlay")
            plt.savefig('train_change/epoch_'+str(iter)+"d_Original_Image_Overlay.png")

            plt.figure()
            plt.axis('off')
            plt.imshow(np.multiply(np.squeeze(test_example),np.squeeze(sess_results)),cmap='gray')
            plt.title('epoch_'+str(iter)+"Generated Overlay")
            plt.savefig('train_change/epoch_'+str(iter)+"e_Generated_Image_Overlay.png")

            plt.close('all')

        # save image if it is last epoch
        if iter == num_epoch - 1:
            train_images,train_labels = shuffle(train_images,train_labels)
            for current_batch_index in range(0,len(train_images),batch_size):
                current_batch = train_images[current_batch_index:current_batch_index+batch_size,:,:,:]
                current_label = np.reshape(train_labels[current_batch_index:current_batch_index+batch_size,:,:,:],(batch_size,-1))
                sess_results = session.run([cost,auto_train,final_softmax],feed_dict={tf_train_dataset:current_batch,tf_train_labels:current_label})

                plt.figure()
                plt.imshow(np.squeeze(current_batch[0,:,:,:]),cmap='gray')
                plt.axis('off')
                plt.title(str(current_batch_index)+"a_Original Image")
                plt.savefig('gif/'+str(current_batch_index)+"a_Original_Image.png")

                plt.figure()
                plt.imshow(np.reshape(np.squeeze(current_label[0,:,:,:],(IMG_WIDTH,IMG_HEIGHT) ) ),cmap='gray')
                plt.axis('off')
                plt.title(str(current_batch_index)+"b_Original Mask")
                plt.savefig('gif/'+str(current_batch_index)+"b_Original_Mask.png")
                
                plt.figure()
                plt.imshow(np.squeeze(np.reshape(sess_results[2][0]),(IMG_WIDTH,IMG_HEIGHT) ),cmap='gray')
                plt.axis('off')
                plt.title(str(current_batch_index)+"c_Generated Mask")
                plt.savefig('gif/'+str(current_batch_index)+"c_Generated_Mask.png")

                plt.figure()
                plt.imshow(np.multiply(np.squeeze(current_batch[0,:,:,:]),np.squeeze(np.reshape(current_label[0](IMG_WIDTH,IMG_HEIGHT) ) )),cmap='gray')
                plt.axis('off')
                plt.title(str(current_batch_index)+"d_Original Image Overlay")
                plt.savefig('gif/'+str(current_batch_index)+"d_Original_Image_Overlay.png")
            
                plt.figure()
                plt.imshow(np.multiply(np.squeeze(current_batch[0,:,:,:]),np.squeeze(np.reshape(sess_results[2][0](IMG_WIDTH,IMG_HEIGHT) ))),cmap='gray')
                plt.axis('off')
                plt.title(str(current_batch_index)+"e_Generated Image Overlay")
                plt.savefig('gif/'+str(current_batch_index)+"e_Generated_Image_Overlay.png")

                plt.close('all')
    if SAVE:
         saver.save(session,SAVE_PATH)

