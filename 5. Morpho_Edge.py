

import cv2 as cv
import numpy as np
import time
#import mahotas as mh;

from skimage.draw import ellipse
from skimage.morphology import label
from skimage.measure import regionprops
from skimage.transform import rotate
from matplotlib import pyplot as plt

Img= cv.imread('messi.jpg',0);

Size = 3;

SE = np.ones((Size,Size));

EImg = cv.dilate(Img, SE)

#EImg = mh.dilate(Img, SE);

MEdge = cv.subtract(EImg, Img);


T_otsu, th_img = cv.threshold(MEdge, 125, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)


#using Sobel edge detector
sobel = np.array([[-1 , 0 , 1] , [-2 , 0 , 2] , [-1 , 0 , 1] ]);

SEdge = cv.filter2D(Img , cv.CV_8U , sobel);

cv.imwrite('SEdge.jpg',SEdge);


plt.gray() 
plt.subplot(151),plt.imshow(Img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(152),plt.imshow(EImg),plt.title('Dilated ')
plt.xticks([]), plt.yticks([])
plt.subplot(153),plt.imshow(MEdge),plt.title('Subtract')
plt.xticks([]), plt.yticks([])

plt.subplot(154),plt.imshow(th_img),plt.title('OTSU')
plt.xticks([]), plt.yticks([])

plt.subplot(155),plt.imshow(SEdge),plt.title('Sobel')
plt.xticks([]), plt.yticks([])

plt.show()