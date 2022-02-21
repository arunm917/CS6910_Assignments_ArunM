# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 11:31:18 2022

@author: Arun Muthukkumaran (CH19D751)
"""

import numpy as np
from keras.datasets import fashion_mnist
#print(dir(fashion_mnist))
train, test = fashion_mnist.load_data()
img, lbl = train
import matplotlib.pyplot as plt

plt.imshow(img[6])
print(lbl[6])
