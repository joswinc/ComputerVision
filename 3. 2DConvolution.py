
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('lena.jpg',1)
 
kernel = np.ones((5,5),np.float32)/25

dst = cv2.filter2D(img,-1,kernel)  #-1 output image depth same as input image depth

GB = cv2.GaussianBlur(img, (5,5), 0.001);

plt.gray() 
plt.subplot(131),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(GB),plt.title('GaussianBlur')
plt.xticks([]), plt.yticks([])
plt.show()


