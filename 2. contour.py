

import numpy as np
import cv2
 
im = cv2.imread('messi.jpg')  
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(im,contours[1],-1,(255,255,255),3)
cv2.imshow("window title", im)
cv2.waitKey()
cv2.destroyAllWindows()