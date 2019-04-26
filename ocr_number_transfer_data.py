# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 19:08:54 2019

@author: m
"""

import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

CHARS_LIST = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'X']
CHARS_NUMS = 11

data_folder = 'F:\\machine-learning\\ML_self-joy\\OCR_exercise\\transfer_training_numbers/'
image_paths = os.listdir(data_folder)

def label_one_hot(label):
  result = np.zeros(CHARS_NUMS)
  if label == 'X':
    result[-1] = 1
  else:
    result[int(label)] = 1
  return result

def create_random_one_data():
  label = random.choice(CHARS_LIST)
  filename = data_folder + ('%s.npy'%(label))
  image_all = np.load(filename).reshape(-1, 784)
  index = random.randint(0, image_all.shape[0]-1)
  image = image_all[index, :]
  return image, label_one_hot(label)

def image_random_rotation(img, fig_show=False):
  img = img.reshape(28, 28)
  rows,cols = img.shape
  angle = random.randint(-10, 10)
  #第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
  M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
  #第三个参数：变换后的图像大小
  res = cv2.warpAffine(img,M,(rows,cols))
  if fig_show:
    plt.subplot(121)
    plt.imshow(img, 'gray')
    plt.subplot(122)
    plt.imshow(res, 'gray')
  return res.reshape(-1)

def create_batch_data(batch_size):
  result_xs = np.empty((batch_size, 784))
  result_ys = np.empty((batch_size, CHARS_NUMS))
  for i in range(batch_size):
    image, label = create_random_one_data()
    result_xs[i, :] = image_random_rotation(image)
    result_ys[i, :] = label
  return result_xs, result_ys
    
  