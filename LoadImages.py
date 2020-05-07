# 2、读取ORL人脸数据库 准备训练数据
import numpy as np
import os
import cv2

def LoadImages(data):
    '''
    加载数据集
    params:
        data:训练集数据所在的目录，要求数据尺寸大小一样
    ret:
        images:[m,height,width]  m为样本数,height为高,width为宽
        names：名字的集合
        labels：标签
    '''
    images = []
    labels = [] 
    names = []

    label = 0
    # 过滤所有的文件夹
    for subDirname in os.listdir(data):
        subjectPath = os.path.join(data, subDirname)
        if os.path.isdir(subjectPath):
            # 每一个文件夹下存放着一个人的照片
            names.append(subDirname)
            for fileName in os.listdir(subjectPath):
                imgPath = os.path.join(subjectPath, fileName)
                img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
                images.append(img)
                labels.append(label)
            label += 1
    images = np.asarray(images)
    labels = np.asarray(labels)
    return images, labels, names

#