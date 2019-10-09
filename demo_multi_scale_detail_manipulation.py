#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   demo_multi_scale_detail_manipulation.py
@time    :   2019/10/08 19:23:07
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :
"""

__author__ = "XiaoY"

#%%
from wls import wls_filter, tone_mapping
from skimage.io import imread, imsave
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
import os


#%%
filepath = "./img/flower.png"
image_name = os.path.split(filepath)[-1]
image_name = os.path.splitext(image_name)[0]

image = imread(filepath)
lab = rgb2lab(image)
luma = lab[:, :, 0]

# wls filter
smooth_luma_0 = wls_filter(luma=luma, lambda_=0.125, alpha=1.2)
smooth_luma_1 = wls_filter(luma=luma, lambda_=0.5, alpha=1.2)

#%%
# fine
val_0 = 25
val_1 = 1
val_2 = 1
exposure = 1.0
saturation = 1.1
gamma = 1.0

lab_fine = tone_mapping(
    lab=lab, luma_0=smooth_luma_0, luma_1=smooth_luma_1,
    val_0=val_0, val_1=val_1, val_2=val_2,
    exposure=exposure, gamma=gamma, saturation=saturation
)
image_fine = lab2rgb(lab=lab_fine)
imsave(fname=os.path.join("./output", image_name + "_fine.png"),
       arr=image_fine)

# medium
val_0 = 1
val_1 = 40
val_2 = 1
exposure = 1.0
saturation = 1.1
gamma = 1.0

lab_medium = tone_mapping(
    lab=lab, luma_0=smooth_luma_0, luma_1=smooth_luma_1,
    val_0=val_0, val_1=val_1, val_2=val_2,
    exposure=exposure, gamma=gamma, saturation=saturation
)
image_medium = lab2rgb(lab=lab_medium)
imsave(fname=os.path.join("./output", image_name + "_medium.png"),
       arr=image_medium)

# coarse
val_0 = 4
val_1 = 1
val_2 = 10
exposure = 1.1
saturation = 1.1
gamma = 1.0

lab_coarse = tone_mapping(
    lab=lab, luma_0=smooth_luma_0, luma_1=smooth_luma_1,
    val_0=val_0, val_1=val_1, val_2=val_2,
    exposure=exposure, gamma=gamma, saturation=saturation
)
image_coarse = lab2rgb(lab=lab_coarse)
imsave(fname=os.path.join("./output", image_name + "_coarse.png"),
       arr=image_coarse)

# combined
image_combined = (image_coarse + image_medium + image_fine) / 3
imsave(fname=os.path.join("./output", image_name + "_combined.png"),
       arr=image_combined)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(2, 2, 1)
ax.imshow(image)
ax.axis("off")
ax = fig.add_subplot(2, 2, 2)
ax.imshow(image_fine)
ax.axis("off")
ax = fig.add_subplot(2, 2, 3)
ax.imshow(image_medium)
ax.axis("off")
ax = fig.add_subplot(2, 2, 4)
ax.imshow(image_coarse)
ax.axis("off")
plt.show()

fig = plt.figure(figsize=(5, 8))
ax = fig.add_subplot(2, 1, 1)
ax.imshow(image)
ax.axis("off")
ax = fig.add_subplot(2, 1, 2)
ax.imshow(image_combined)
ax.axis("off")
plt.show()


#%%
