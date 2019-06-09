
import cv2 as cv
import numpy as np
from skimage import transform


Img = cv.imread('messi.jpg',0)

h,w = Img.shape[:2]

res = transform.resize(Img, (round(h*0.5),round(w*0.5)))
cv.imshow('Original', Img)
cv.waitKey(0)

cv.imshow('resize', res)
cv.waitKey(0)
cv.destroyAllWindows()