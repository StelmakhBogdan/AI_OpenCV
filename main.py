import cv2

img = cv2.imread('images/fancy.jpeg')

new_img = cv2.resize(img, (img.shape[1] // 2, img.shape[2] // 2))

cv2.imshow('Photo', img)
cv2.waitKey(1000)

