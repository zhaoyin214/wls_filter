#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   zero_one_norm.py
@time    :   2019/10/08 16:22:03
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :   max-min normalization (zero-one)
"""

__author__ = "XiaoY"

#%%
import numpy as np


#%%
def zero_one_norm(input):

    max_ = np.max(a=input)
    min_ = np.min(a=input)

    return (input - min_) / (max_ - min_)

