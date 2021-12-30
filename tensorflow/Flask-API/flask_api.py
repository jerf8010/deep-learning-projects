# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 00:25:11 2021

@author: jerf
"""

from tensorflow.keras.datasets import fashion_mnist
from imageio import imwrite

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

for i in range(5):
    imwrite('D://jose/cursos/cursos udemy/deep learning projects/tensorflow/Flask-API/uploads/{}.png'.format(i), X_test[i])