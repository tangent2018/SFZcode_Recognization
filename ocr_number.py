# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:37:47 2019

@author: m
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from trainIDCard import *
from genIDCard import *

#image size: 1080, 1440, 3
def getSFZ(img,threshold_w = [15,1200],threshold_h = [0,200],threshold_data = [50,200],threshold_xmin=[400,1400],threshold_ymin=[800,1000]):
    """
    用于定位￥符号位置
    img : 原始cv读取的图像
    threshold_w : ￥宽度范围
    threshold_h : ￥高度范围
    threshold_xmin : ￥出现的位置范围
    """
    img_gray = img[:,:,0]
    ret, binary = cv2.threshold(img_gray,115,255,cv2.THRESH_BINARY)
    cnts, hierarchy = cv2.findContours(binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    positions = []
    for index,c in enumerate(cnts):
        x,y,w,h = cv2.boundingRect(c)
        xmin,ymin,xmax,ymax = x,y,x+w,y+h
        if threshold_xmin[0]<xmin<threshold_xmin[1] and threshold_ymin[0]<ymin<threshold_ymin[1]:
            if threshold_h[0]<h<threshold_h[1] and threshold_w[0]<w<threshold_w[1] :
                data = np.mean(img[ymin:ymax,xmin:xmax])
                if threshold_data[0]<data<threshold_data[1]:
                    positions.append([xmin,ymin,xmax,ymax])
    return positions

def rectangleImg(img,positions):
    for position in positions:
        xmin,ymin,xmax,ymax = position
        xmed,ymed = (xmin+xmax)//2,(ymin+ymax)//2
        cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0),5)
    return img
    
def united_pos(positions):
  #xmin取所有xmin中最小的 xmax取所有xmax中最大的 
  #ymin取所有ymin中与ycenter最接近的 ymax取所有ymax中与ycenter最接近的 ycenter是所有ymin和ymax的平均值
  positions = np.array(positions)
  xmin = np.min(positions[:, 0])
  xmax = np.max(positions[:, 2])
  ycenter = np.mean(positions[:, (1,3)])
  ymin_arg = np.argmax(ycenter-positions[:, 1])
  ymin = positions[ymin_arg, 1]
  ymax_arg = np.argmax(positions[:, 3]-ycenter)
  ymax = positions[ymax_arg, 3]
  return xmin,ymin,xmax,ymax
 
def read_img(img_path, fig_show=False):
  img = cv2.imread(img_path)
  img_frame = img.copy()
  positions = getSFZ(img)
  position_u = [united_pos(positions)]
  img_frame = rectangleImg(img_frame,position_u)
  img_code = img[position_u[0][1]:position_u[0][3]+1, position_u[0][0]:position_u[0][2]+1, 0]
  img_resize = cv2.resize(img_code, (256, 32))
  ret, img_binary = cv2.threshold(img_resize,70,255,cv2.THRESH_BINARY)
  img_minus = np.abs(img_binary-255)
  # 显示结果
  if fig_show:
    f,(ax1,ax2) = plt.subplots(2,1,figsize=(16,12))
    ax1.imshow(cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB)) 
    ax2.imshow(img_minus, 'gray') 
  
  return img_minus

def X_iterator():
  for i in range(3):
    img_path = 'F:/machine-learning/ML_self-joy/OCR_exercise/shenfenzhengX3/shenfenzheng0%d.jpg'%(i+1)
    yield read_img(img_path)

def vec2num(vec):
  vec = np.argmax(vec.reshape((MAX_CAPTCHA, CHAR_SET_LEN)),1)
  vec = [str(num if num!= 10 else 'X') for num in vec]
  vec = ''.join(vec)
  return vec
#%%
model_dir = './models/'
IMAGE_HEIGHT, IMAGE_WIDTH = 32, 256
MAX_CAPTCHA = 18
CHAR_SET_LEN = 11
with tf.Graph().as_default() as g:
  X_input = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH])
  keep_prob = tf.placeholder(tf.float32) # dropout
  train_phase = tf.placeholder(tf.bool)
  
  with tf.device('/gpu:0'):
    output = crack_captcha_cnn(X_input, train_phase, keep_prob)

  saver = tf.train.Saver()
  config = tf.ConfigProto(allow_soft_placement=True)
  y_predict_ret = []
  with tf.Session(config=config) as sess:
    model_checkpoint_path = model_dir+'crack_capcha.ckpt-1500'
    saver.restore(sess, model_checkpoint_path)
    for x_img in X_iterator():
      plt.imshow(x_img, 'gray')
      print (x_img.shape)
      x_img = x_img.reshape((1, IMAGE_HEIGHT*IMAGE_WIDTH))
      y_output = sess.run(output, feed_dict={X_input: x_img, keep_prob: 1, train_phase:False})
      y_predict = vec2num(y_output)
      print (y_predict)
      y_predict_ret.append(y_predict)
    
    image_test, text, vec = gen_id_card().gen_image()
    plt.imshow(image_test, 'gray')
    print (image_test.shape)
    image_test = image_test.reshape((1, IMAGE_HEIGHT*IMAGE_WIDTH))
    y_output = sess.run(output, feed_dict={X_input: image_test, keep_prob: 1, train_phase:False})
    y_predict = vec2num(y_output)
    print (y_predict)
    y_predict_ret.append(y_predict)

