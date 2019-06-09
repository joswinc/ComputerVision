

import cv2 as cv
from matplotlib import pyplot as plt
CImg = cv.imread('lena.jpg')

GImg = cv.cvtColor(CImg, cv.COLOR_BGR2GRAY)
histOriginal = cv.calcHist([GImg],[0],None,[256],[0,256])

HEImg = cv.equalizeHist(GImg)
histProcessed = cv.calcHist([HEImg],[0],None,[256],[0,256])

f, axs = plt.subplots(2,2,figsize=(15,15))

axs[0][0].plot(histOriginal)
axs[0][0].set_title('Original Histogram')

axs[0][1].plot(histProcessed)
axs[0][1].set_title('Equalized Histogram')


axs[1][0].imshow(GImg, cmap = 'gray')
axs[1][0].set_title('Original Image')

axs[1][1].imshow(HEImg, cmap = 'gray')

axs[1][1].set_title('Image after HEq')

plt.show()