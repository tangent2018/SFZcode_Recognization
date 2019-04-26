# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:37:47 2019

@author: m
"""
import numpy as np
import tensorflow as tf
import ocr_number_MNIST

# refernce argument values
PIXEL_DEPTH = 255

# test with test data given by mnist_data.py
def test(model_directory):
  with tf.Graph().as_default() as g:
    #load graph and tensor
    meta_filename = model_directory+'.meta'
    saver = tf.train.import_meta_graph(meta_filename)
    input_x = g.get_tensor_by_name('Placeholder:0')
    is_training = g.get_tensor_by_name('MODE:0')
    fco11 = g.get_tensor_by_name('fco11/BiasAdd:0')
    print ('Getting tensor.........................')
    print ('Tensor input_x, name: <Placeholder:0>', input_x.shape)
    print ('Tensor is_training, name: <MODE:0>', is_training.shape)
    print ('Tensor fco11, name: <fco11/BiasAdd:0>', fco11.shape)
    
  with tf.Session(graph = g) as sess:
    # Restore variables from disk
    saver.restore(sess, model_directory)
    
    y_ret = []
    # Loop over all batches
    for xs in ocr_number_MNIST.next_img():
        xs = (xs - (PIXEL_DEPTH / 2.0) / PIXEL_DEPTH)  # make zero-centered distribution as in mnist_data.extract_data()
        y_predict = sess.run(fco11, feed_dict={input_x: xs, is_training: False})
        y_ret.append(y_predict)
        
  return y_ret
 
def array2str(array):
  num_list = [str(num if num!=10 else 'X') for num in array]
  return ''.join(num_list)

def correct_code(code):
  #验证校验位
  code_list = [int(num if num!='X' else '10') for num in code]
  correct_np = np.array([7,9,10,5,8,4,2,1,6,3,7,9,10,5,8,4,2])
  code_np = np.array(code_list[:17])
  val_code = np.dot(code_np,correct_np)%11
  val_code_affine = ['1','0','10','9','8','7','6','5','4','3','2'] #'X'对应'10'
  val_code_real = val_code_affine[val_code]
  if val_code_real != str(code_list[-1]):
    print ('for SFZcode: %s' % (code))
    print ('val_code_real is %s, val_code_predict is %s' % (val_code_real, code_list[-1] if code_list[-1]!=10 else 'X'))
    return False
  return True

model_directory = "./transfer_model"

cycle = 1
while True:
  correct_flag = True
  print ('Cycle %d starts.' % (cycle))
  y_ret = test(model_directory+'/model.ckpt')
  predict_ret = []
  for y_predict in y_ret:
    if correct_flag:
      predict = np.argmax(y_predict, axis=1)
      predict_array = array2str(predict)
      if not correct_code(predict_array):
        correct_flag = False
        break    
      predict_ret.append(predict_array)
  if correct_flag:
    print ('End in cycle %d.' % (cycle))
    for code in predict_ret:
      print (code)
    break
  else:
    cycle += 1

