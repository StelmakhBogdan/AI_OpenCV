import cv2
import numpy as np

# try to do binary img for easy clustering material

img = cv2.imread('images/fancy.jpeg')

resize_img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

bw_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
new_img = cv2.Canny(bw_img, 90, 90)

kernel = np.ones((5, 5), np.uint8)

dilate = cv2.dilate(new_img, kernel,  iterations=1)
erode = cv2.erode(dilate, kernel,  iterations=1)

cv2.imshow('Photo', erode)
cv2.waitKey(0)

