# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 11:31:18 2022

@author: Arun Muthukkumaran (CH19D751)
"""

import numpy as np
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import time
#print(dir(fashion_mnist))
train, test = fashion_mnist.load_data()
img, lbl = train

unique_classes = np.unique(lbl)
img_store = []

for i in range (0,unique_classes.size):
    lbl_index = np.where(lbl == [i])
    lbl_index_arr = np.array(lbl_index)
    img1 = img[lbl_index_arr[0,0]]
    img_store.append(img1)
    plt.imshow(img1)
    
for i in img_store:    
    
    plt.imshow(i)
    plt.show()
    time.sleep(0.01)
