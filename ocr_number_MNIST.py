# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:37:47 2019

@author: m
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from trainIDCard import *
from genIDCard import *

#image size: 1080, 1440, 3
def getSFZcode(img,threshold_w = [15,1200],threshold_h = [0,200],threshold_data = [50,150],threshold_xmin=[400,1400],threshold_ymin=[800,1000]):
    """
    用于定位￥符号位置
    img : 原始cv读取的图像
    threshold_w : ￥宽度范围
    threshold_h : ￥高度范围
    threshold_xmin : ￥出现的位置范围
    """
    img_gray = img[:,:,0]
    #ret, binary = cv2.threshold(img_gray,115,255,cv2.THRESH_BINARY)
    ret, binary = cv2.threshold(img_gray,80,255,cv2.THRESH_BINARY)
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
  #按照positions在img画框
    for position in positions:
        xmin,ymin,xmax,ymax = position
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

def split_code(img,threshold_w = [10,50],threshold_h = [30,74],threshold_data = [50,200]):
    # 分割各个数字
    cnts, hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    positions = []
    for index,c in enumerate(cnts):
        x,y,w,h = cv2.boundingRect(c)
        xmin,ymin,xmax,ymax = x,y,x+w,y+h
        if threshold_h[0]<h<threshold_h[1] and threshold_w[0]<w<threshold_w[1] :
            data = np.mean(img[ymin:ymax,xmin:xmax])
            if threshold_data[0]<data<threshold_data[1]:
                positions.append([xmin,ymin,xmax,ymax])
    return positions

def positions_sort_fn(positions):
  #将positions从小到大排序
  sort_dict = {}
  sort_pos0 = []
  sort_ret = []
  for pos in positions:
    sort_dict[pos[0]] = positions.index(pos)
    sort_pos0.append(pos[0])
  sort_pos0.sort()
  for pos0 in sort_pos0:
   sort_ret.append(positions[sort_dict[pos0]])
  return sort_ret

def create_num(img, positions_sort):
  #根据positions_sort将img转化为18*28*28的np.array
  ret = None
  for position in positions_sort:
    num_img = img[position[1]:position[3]+1, position[0]:position[2]+1]
    img_h, img_w = num_img.shape
    ret_img_w = ret_img_h = int(img_h*1.3)
    ret_img = np.ones((ret_img_h, ret_img_w))*255
    start_up = int((ret_img_h-img_h)/2)
    start_left = int((ret_img_w-img_w)/2)
    ret_img[start_up:start_up+img_h, start_left:start_left+img_w] = num_img
    ret_img_resize = cv2.resize(ret_img, (28,28))[np.newaxis, :]
    if ret is None:
      ret = ret_img_resize
    else:
      ret = np.concatenate((ret, ret_img_resize))
  return ret
  
def imshow_nums_img(nums_img):
  #连续图示nums_img
  nums_img_len = len(nums_img)
  nums_img = nums_img.reshape(nums_img_len, 28, 28)
  plt.figure(figsize=(7,7))
  for i in range(nums_img_len):
    plt.subplot(1,nums_img_len,i+1)
    plt.imshow(nums_img[i,:,:],'gray')

def change2MNIST(imgs):
  #将数据转化为对应MNIST的数据 数据由0-255变为0-1，再-1取绝对值，最后，将28X28展平
  return np.abs(imgs/255-1).reshape(imgs.shape[0], -1)

def create_img_data(img_path):
  #输入身份证图片地址，输出一个代表身份证号码的(18, 784)的array
  img = cv2.imread(img_path)
  img = cv2.resize(img, (1440,1080))
  positions = getSFZcode(img)
  position_u = united_pos(positions)
  img_code = img[position_u[1]:position_u[3]+1, position_u[0]:position_u[2]+1, 0]
  ret, img_binary = cv2.threshold(img_code,70,255,cv2.THRESH_BINARY)
  positions = split_code(img_binary)
  positions_sort = positions_sort_fn(positions)
  nums_img = create_num(img_binary, positions_sort)
  nums_imgs = change2MNIST(nums_img)
  assert nums_imgs.shape[0]==18, 'image_data数量为：%d，应为18'%(nums_imgs.shape[0])
  return nums_imgs
 
def next_img():
  file_top = 'F:/machine-learning/ML_self-joy/OCR_exercise/shenfenzhengX3/'
  for _, _, img_paths in os.walk(file_top):
    for img_path in img_paths:
      yield create_img_data(file_top+img_path)

if __name__ == '__main__':
  img_path = 'F:/machine-learning/ML_self-joy/OCR_exercise/shenfenzhengX3/310110198710192017.jpg'
  image_data = create_img_data(img_path)
  print (image_data.shape[0])
  imshow_nums_img(image_data)

