# python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5s.pt
# python detect.py --weights yolov5/runs/train/exp5/weights/best.pt --img 640 --conf 0.4 --source test\test5.jpg --save-crop

import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import pytesseract

img = cv2.imread('yolov5/runs/detect/exp13/crops/license plate/test3.jpg')
original = img.copy()

""" rotate_img = imutils.rotate(img, 5)
plt.imshow(rotate_img)
plt.show()  """

""" up_width = 500
up_height = 200
up_points = (up_width, up_height)
resized_up = cv2.resize(img, up_points, interpolation= cv2.INTER_LINEAR) """
""" plt.imshow(resized_up)
plt.show() """

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
""" plt.imshow(gray, cmap='gray')
plt.show() """

blur = cv2.GaussianBlur(gray, (3,3), 0)
canny = cv2.Canny(blur, 120, 255, 1)

""" (thresh, binary) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
 """""" plt.imshow(binary, cmap='gray')
plt.show() """

""" kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3) """
""" plt.imshow(thre_mor, cmap='gray')
plt.show() """

""" cnts = cv2.findContours(binary.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

clone = img.copy()
cv2.drawContours(clone, cnts, 0, (0, 0, 255), 1)
clone = imutils.resize(clone, width=400)

cv2.imshow("All Contours", clone)
cv2.waitKey(0) """

cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

min_area = 100
image_number = 0
for c in cnts:
    area = cv2.contourArea(c)
    if area > min_area:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,0), 2)
        ROI = original[y:y+h, x:x+w]
        cv2.imwrite("ROI_{}.png".format(image_number), ROI)
        image_number += 1

cv2.imshow('blur', blur)
cv2.imshow('canny', canny)
cv2.imshow('image', img)
cv2.waitKey(0) 

pic = cv2.imread('ROI_2.png')

gray2 = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
(thresh, binary) = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cnts = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

original_2 = pic.copy()
cv2.drawContours(original_2, cnts, 0, (0, 0, 0), 1)

""" cv2.imshow('ori',pic)
cv2.imshow('gay', gray2)
cv2.imshow('bi', binary)
cv2.imshow("All Contours", original_2) """
""" cv2.waitKey(0) """
 
# Adding custom options
# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\dewdu\AppData\Local\Tesseract-OCR'
""" lang="-l tha --psm 6"
res = pytesseract.image_to_string(original_2,config=lang)
print(res)  """

""" reader = easyocr.Reader(['th'])
result = reader.readtext(original_2)
print(result) """

"""plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))

bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
edged = cv2.Canny(bfilter, 30, 200) #Edge detection
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))

keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break

mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0,255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)

plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.title("BW")
plt.show()

(x,y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.show() """

""" reader = easyocr.Reader(['th'])
result = reader.readtext(binary)
print(result)  """

""" text = result[0][-2]
font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
plt.show() """