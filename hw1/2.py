import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # open input image
    imageFileName = 'img/bee.jpg'
    image = cv2.imread(imageFileName)

    