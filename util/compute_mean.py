#-*- coding : utf-8 -*-
# coding: utf-8

import numpy as np
import random
from osgeo import gdal
from PIL import Image
import os

"""
    使用opencv和numpy计算图像数据集的均值和方差
    随机挑选CNum张图片,进行按通道计算均值mean和标准差std
    先将像素从0-255归一化至 0-1 再计算
"""
####----------------------------------  GF-2 --------------------------------------####
# train_txt_path = './dataset/jinzhou/B_512/B_512.txt'

# CNum = 289     # 挑选多少图片进行计算 GF-2:100 

# img_h, img_w = 512, 512  # GF-2:512
# imgs = np.zeros([1, 4, img_h, img_w])
# means, stdevs = [], []

# with open(train_txt_path, 'r') as f:
#     lines = f.readlines()
#     random.shuffle(lines)   # shuffle , 随机挑选图片

#     for i in range(CNum):
#         img_path = lines[i].rstrip().split()[0]
#         img = gdal.Open(img_path).ReadAsArray()
#         img = img[np.newaxis, :, :, :]
#         imgs = np.concatenate((imgs, img), axis=0)
#         # print(i)

# imgs = np.delete(imgs,np.s_[0],axis=0)
# #imgs = imgs.astype(np.float64)/255.

# for i in range(4):#四个通道
#     pixels = imgs[:,i,:,:].ravel()  # 拉成一行
#     pixels = (pixels-np.min(pixels))/(np.max(pixels)-np.min(pixels)) # 归一化
#     means.append(np.mean(pixels))
#     stdevs.append(np.std(pixels))

# # means.reverse() # BGR --> RGB
# # stdevs.reverse()

# print("normMean = {}".format(means))
# print("normStd = {}".format(stdevs))
# print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))

####----------------------------------  LEVIR-CD --------------------------------------####
train_txt_path = './WHU512/train/A/A.txt' 
CNum = 1260
img_h, img_w = 512, 512  # GF-2:512
imgs = np.zeros([1, 3, img_h, img_w])
means, stdevs = [], []

with open(train_txt_path, 'r') as f:
    lines = f.readlines()
    random.shuffle(lines)   # shuffle , 随机挑选图片

    for i in range(CNum):
        img_path = lines[i].rstrip().split()[0]
        img = np.asarray(Image.open(os.path.join('./WHU512/train/B/', img_path)).convert('RGB')).reshape(3, 512, 512)
        img = img[np.newaxis, :, :, :]
        imgs = np.concatenate((imgs, img), axis=0)

imgs = np.delete(imgs,np.s_[0],axis=0)

for i in range(3):
    pixels = imgs[:,i,:,:].ravel()  # 拉成一行
    pixels = (pixels-np.min(pixels))/(np.max(pixels)-np.min(pixels)) # 归一化
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))