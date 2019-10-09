#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   sigmoid.py
@time    :   2019/09/30 12:15:25
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :
"""

__author__ = "XiaoY"


#%%
import numpy as np


#%%
def sigmoid(x, a):

    # sigmoid
    y = 1 / (1 + np.exp(- a * x)) - 0.5

    # rescale
    y *= 0.5 / (1 / (1 + np.exp(- a * 0.5)) - 0.5)

    return y


#%%
if __name__ == "__main__":
    x = np.random.randint(- 10, 10, (5, 5))
    print(sigmoid(x, 0.5))

#%%
