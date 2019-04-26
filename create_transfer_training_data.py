# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:37:47 2019

@author: m
"""
#使用shenfenzhengX3文件夹内的图像，生成数字样本
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from ocr_number_MNIST import create_img_data, imshow_nums_img

savedir = './transfer_training_numbers/'

file_top = 'F:/machine-learning/ML_self-joy/OCR_exercise/shenfenzhengX3/'
nums_array_dict = {}
index_list = ['0','1','2','3','4','5','6','7','8','9','X']
for index in index_list:
  nums_array_dict[index] = []
for _, _, img_paths in os.walk(file_top):
  for img_path in img_paths:
    num_name_list = list(img_path.split('.')[0])
    data_array = create_img_data(file_top+img_path)
    for i in range(len(num_name_list)):
      nums_array_dict[num_name_list[i]].append(data_array[i,:])

for key in nums_array_dict.keys():
  array_result = nums_array_dict[key].pop().reshape(1, -1)
  while nums_array_dict[key] != []:
    array_result = np.concatenate((array_result, nums_array_dict[key].pop().reshape(1, -1)), axis=0)
  filename = savedir+('%s.npy'%(key))
  np.save(filename, array_result)


