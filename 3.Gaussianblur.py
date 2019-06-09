

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from skimage import color, data, restoration


astro = color.rgb2gray(data.chelsea())
#astro = color.rgb2gray(cv.imread('D:\\Images\\blurred-buildings\\moon.jpg',1))
from scipy.signal import convolve2d as conv2




GB = cv.GaussianBlur(astro, (3,3), 0.1);

p=0.05
GDB = p*astro - (1-p)*GB;





plt.gray()

plt.subplot(131),plt.imshow(astro),plt.title('original')
plt.xticks([]), plt.yticks([])


plt.subplot(132),plt.imshow(GDB),plt.title('blurred')
plt.xticks([]), plt.yticks([])


plt.subplot(133),plt.imshow(GDB),plt.title('Deblurred')
plt.xticks([]), plt.yticks([])

plt.show()
