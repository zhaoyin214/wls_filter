#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   tone_mapping.py
@time    :   2019/10/08 12:35:23
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :   tone manipulation
"""

__author__ = "XiaoY"


#%%
from utils import sigmoid, zero_one_norm
import numpy as np


#%%
def tone_mapping(lab, luma_0, luma_1,
                 val_0, val_1, val_2,
                 exposure=1, gamma=1.1, saturation=1):
    """
    tone maps an image in the CIELAB color space

    arguments:
        lab (2-dim array, required) - the image in CIELAB color space
        luma_0, luma_1 (2-dim array, required) - smoothed versions of L of LAB
        val_0-val_2 (float, [-1, 1]) - compression/expansion params
        exposure (float, [0, inf))
        gamma (float, (0, 1])
        saturation (float, [0, inf))

    return:
        out
    """
    lab = lab.copy()
    luma = lab[:, :, 0]

    detail_0 = luma - luma_0
    if val_0 > 0:
        detail_0 = sigmoid(x=detail_0 / 100, a=val_0) * 100
    elif val_0 < 0:
        detail_0 = (1 + val_0) * detail_0
    else:
        pass

    detail_1 = luma_0 - luma_1
    if val_1 > 0:
        detail_1 = sigmoid(x=detail_1 / 100, a=val_1) * 100
    elif val_1 < 0:
        detail_1 = (1 + val_1) * detail_1
    else:
        pass

    base = exposure * luma_1
    if val_2 > 0:
        base = sigmoid(x=(base - 56) / 100, a=val_2) * 100 + 56
    elif val_2 < 0:
        base = (1 + val_2) * (base - 56) + 56
    else:
        pass

    if gamma == 1:
        luma = base + detail_0 + detail_1
    else:
        base_max = np.max(base)
        luma = zero_one_norm(input=base) ** gamma * base_max + \
            detail_0 + detail_1

    lab[:, :, 0] = luma
    # lab[:, :, 0] = np.clip(a=luma, a_min=0, a_max=100)
    if saturation != 0:
        lab[:, :, 1 :] *= saturation
        # lab[:, :, 1 :] = (lab[:, :, 1 :] - 128) * saturation + 128
    else:
        pass

    return lab

#%%
if __name__ == "__main__":
    from skimage.io import imread
    from skimage.color import rgb2lab, lab2rgb
    import matplotlib.pyplot as plt
    import numpy as np
    from wls import wls_filter

    exposure = 1.0
    saturation = 1.1
    gamma = 1.0

    filepath = "./img/flower.png"
    image = imread(filepath)
    lab = rgb2lab(image)
    luma = lab[:, :, 0]

    smooth_luma_0 = wls_filter(luma=luma, lambda_=0.125, alpha=1.2)
    smooth_luma_1 = wls_filter(luma=luma, lambda_=0.5, alpha=1.2)
    lab = tone_mapping(
        lab=lab, luma_0=smooth_luma_0, luma_1=smooth_luma_1,
        val_0=25, val_1=1, val_2=1,
        exposure=exposure, gamma=gamma, saturation=saturation
    )
    image_out = lab2rgb(lab=lab)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(image)
    ax.axis("off")
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(image_out)
    ax.axis("off")
    plt.show()


#%%
