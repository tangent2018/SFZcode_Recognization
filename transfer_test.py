# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 10:07:31 2019

@author: m
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

batch_xs = np.load('batch_xs.npy')
batch_ys = np.load('batch_ys.npy')

def find_tensor_by_name(graph, name):
  # 在graph中输入tensor可能包含的name，得到一个包含这个name的operations的list
  ret = []
  for operation in graph.get_operations():
    if str(operation).find(name) != -1:
      print (operation)
      ret.append(operation)
  return ret

ckpt_filename = "./tensorflow_mnist_cnn_master/model/model.ckpt"
meta_filename = './tensorflow_mnist_cnn_master/model/model.ckpt.meta'

sess = tf.Session()
#with tf.Session() as sess:
# 先加载图和变量
saver = tf.train.import_meta_graph(meta_filename)
saver.restore(sess, ckpt_filename)

# 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
graph = tf.get_default_graph()

input_x = graph.get_tensor_by_name('Placeholder:0')
input_y = graph.get_tensor_by_name('Placeholder_1:0')
is_training = graph.get_tensor_by_name('MODE:0')
feed_dict = {input_x: batch_xs, input_y: batch_ys, is_training: False}

dropout3 = graph.get_tensor_by_name('dropout3/dropout/mul:0')
fco = graph.get_tensor_by_name('fco/BiasAdd:0')
output2 = sess.run(fco, feed_dict)

