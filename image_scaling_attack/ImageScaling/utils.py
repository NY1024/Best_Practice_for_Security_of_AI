"""
Author: Yufei Chen
Mail: yfchen@sei.xjtu.edu.cn
"""

import os
import sys
import cv2
import numpy as np

def imgLoader(imgPath, color_flag=cv2.IMREAD_COLOR):
    #使用opencv从指定路径加载图像，以numpy矩阵的格式返回图像的数据，shape为（w,h,c）
    try:
        img = cv2.imread(imgPath, color_flag)
        height, width, *channel = img.shape
        if not channel:
            img = img.reshape((height, width, 1))
        return img
    except:
        print('Fail to load the image %s' %imgPath)

def imgSaver(imgPath, img):
    #使用opencv将图像保存到指定位置
    try:
        img = cv2.imwrite(imgPath, img)
        return img
    except:
        print('Fail to save the image as %s' %imgPath)


def test():
    pass

