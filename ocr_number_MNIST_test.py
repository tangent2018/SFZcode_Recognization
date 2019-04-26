# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:37:47 2019

@author: m
"""
import numpy as np
import tensorflow as tf
import ocr_number_MNIST
from tensorflow_mnist_cnn_master import cnn_model

# refernce argument values
PIXEL_DEPTH = 255

# test with test data given by mnist_data.py
def test(model_directory):
    # tf Graph input
    x = tf.placeholder(tf.float32, [None, 784])
    is_training = tf.placeholder(tf.bool, name='MODE')
    y = cnn_model.CNN(x, is_training=is_training)
    
    # Add ops to save and restore all the variables
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})

    # Restore variables from disk
    saver = tf.train.Saver()
    saver.restore(sess, model_directory)
    
    y_ret = []
    # Loop over all batches
    for xs in ocr_number_MNIST.next_img():
        xs = (xs - (PIXEL_DEPTH / 2.0) / PIXEL_DEPTH)  # make zero-centered distribution as in mnist_data.extract_data()
        y_predict = sess.run(y, feed_dict={x: xs, is_training: False})
        y_ret.append(y_predict)
        
    return y_ret
 
def array2str(array):
  num_list = [str(num) for num in array]
  return ''.join(num_list)

def correct_code(code):
  code_list = [int(num) for num in code]
  correct_np = np.array([7,9,10,5,8,4,2,1,6,3,7,9,10,5,8,4,2])
  code_np = np.array(code_list[:17])
  val_code = np.dot(code_np,correct_np)%11
  val_code_affine = ['1','0','X','9','8','7','6','5','4','3','2']
  val_code_real = val_code_affine[val_code]
  if val_code_real != str(code_list[-1]):
    print ('val_code_real is %s, val_code_predict is %s' % (val_code_real, code_list[-1]))
  if val_code_real == 'X':
    code = code[:17]+'X'
  return code
    
if __name__ == '__main__':
  model_directory = "./tensorflow_mnist_cnn_master/model"
  y_ret = test(model_directory+'/model.ckpt')
  predict_ret = []
  for y_predict in y_ret:
    modify_add1 = np.zeros_like(y_predict)
    modify_add1[:,1] = 4
    y_predict += modify_add1
    predict = np.argmax(y_predict, axis=1)
    predict_array = array2str(predict)
    predict_correct = correct_code(predict_array)
    predict_ret.append(predict_correct)
