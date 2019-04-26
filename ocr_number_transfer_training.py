# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 10:07:31 2019

@author: m
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import numpy as np
import os
from ocr_number_transfer_data import create_batch_data

def find_tensor_by_name(graph, name):
  # 在graph中输入tensor可能包含的name，得到一个包含这个name的operations的list
  ret = []
  for operation in graph.get_operations():
    if str(operation).find(name) != -1:
      print (operation)
      ret.append(operation)
  return ret
  
# Data Training

# restoring data info
ckpt_filename = "./tensorflow_mnist_cnn_master/model/model.ckpt"
meta_filename = './tensorflow_mnist_cnn_master/model/model.ckpt.meta'
# saving data info
MODEL_DIRECTORY = "transfer_model/model.ckpt"
LOGS_DIRECTORY = "transfer_logs/train"

# Params for Train
training_epochs = 1000
train_batch_size = 256

with tf.Graph().as_default() as g:
  #load graph and tensor
  saver_restore = tf.train.import_meta_graph(meta_filename)
  input_x = g.get_tensor_by_name('Placeholder:0')
  input_y = g.get_tensor_by_name('Placeholder_1:0')#
  is_training = g.get_tensor_by_name('MODE:0')
  dropout3 = g.get_tensor_by_name('dropout3/dropout/mul:0')
  weight_test = g.get_tensor_by_name('conv2/weights:0')
  print ('Getting tensor.........................')
  print ('Tensor input_x, name: <Placeholder:0>', input_x.shape)
  print ('Tensor is_training, name: <MODE:0>', is_training.shape)
  print ('Tensor dropout3, name: <dropout3/dropout/mul:0>', dropout3.shape)
  
  #stop the backpropogation
  dropout3 = tf.stop_gradient(dropout3,name='stop_gradient')
  
  y_ = tf.placeholder(tf.float32, [None, 11], name = 'input_y_11') #answer
  
  # Predict
  with slim.arg_scope([slim.fully_connected]):
    y = slim.fully_connected(dropout3, 11, activation_fn=None, normalizer_fn=None, scope='fco11')
  
  # Get loss of model
  with tf.name_scope("LOSS_transfer"):
    loss = tf.losses.softmax_cross_entropy(logits=y, onehot_labels=y_)
  
  # Create a summary to monitor loss tensor
  tf.summary.scalar('loss', loss)
  
  with tf.name_scope("ADAM"):
    step = tf.Variable(0, trainable=False, name = 'global_step')
    learning_rate = tf.train.exponential_decay(
        1e-4,  # Base learning rate.
        step * train_batch_size,  # Current index into the dataset.
        decay_steps = 5000,  # Decay step.
        decay_rate = 0.95)  # Decay rate.
        # Use simple momentum for the optimization.
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=step)

  # Create a summary to monitor learning_rate tensor
  tf.summary.scalar('learning_rate', learning_rate)
  
  # Get accuracy of model
  with tf.name_scope("ACC_transfer"):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
  # Create a summary to monitor accuracy tensor
  tf.summary.scalar('acc', accuracy)
  
  # Merge all summaries into a single op
  merged_summary_op = tf.summary.merge_all()

  saver_save = tf.train.Saver()
  
# Add ops to save and restore all the variables
with tf.Session(graph = g) as sess: 
  sess.run(tf.global_variables_initializer())
  saver_restore.restore(sess, ckpt_filename)
  
  # op to write logs to Tensorboard
  summary_writer = tf.summary.FileWriter(LOGS_DIRECTORY, graph=tf.get_default_graph())
  
  # Loop for epoch
  for epoch in range(training_epochs):
      train_data, train_labels = create_batch_data(train_batch_size)
      input_y_dummy = np.zeros((train_labels.shape[0], 10))#用于填充input_y的dummy
      
      # Run optimization op (backprop), loss op (to get loss value)
      # and summary nodes
      _, train_accuracy, summary, step_, loss_ = \
                sess.run([train_op, accuracy, merged_summary_op, step, loss], 
                feed_dict={input_x: train_data, y_: train_labels, input_y: input_y_dummy, is_training: True})
  
      # Write logs at every iteration
      summary_writer.add_summary(summary, global_step = step_)
  
      # Display logs
      if step_ % 10 == 0:
        print('step: %04d, loss: %f, training accuracy: %.5f' % (step_ + 1, loss_, train_accuracy))
        '''
        if train_accuracy == 1:
          save_path = saver_save.save(sess, MODEL_DIRECTORY)
          print ("Model updated and saved in file: %s" % save_path)
          break
        '''

  print("Optimization Finished!")
  
  #Data validating
  val_batch_size = 500
  val_data, val_labels = create_batch_data(val_batch_size)
  y_final = sess.run(y, feed_dict={input_x: val_data, is_training: False})
  correct_prediction = np.equal(np.argmax(y_final, 1), np.argmax(val_labels, 1))
  val_acc = np.sum(correct_prediction) / val_batch_size
  
  print("val_acc for the stored model: %.3f" % (val_acc))


