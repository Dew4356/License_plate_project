import numpy as np
import cv2
import imutils

image = cv2.imread("./test/test3.jpg")
image = imutils.resize(image,width = 400)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
cv2.imshow("Original", image)

cnts = cv2.findContours(gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

clone = image.copy()
cv2.drawContours(clone, cnts, -1, (0, 255, 0), 2)

cv2.imshow("All Contours", clone)
cv2.waitKey(0)