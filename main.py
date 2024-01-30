import cv2
import numpy as np

import imutils

# from helpers.rotate import rotate

# try to do binary img for easy clustering material

# img = cv2.imread('images/fancy.jpeg')

# resize_img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

# bw_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
# new_img = cv2.Canny(bw_img, 90, 90)

# kernel = np.ones((5, 5), np.uint8)

# dilate = cv2.dilate(new_img, kernel,  iterations=1)
# erode = cv2.erode(dilate, kernel,  iterations=1)

# cv2.imshow('Photo', erode)
# cv2.waitKey(0)


# photo = np.zeros((600, 450, 3), dtype='uint8')

# BGR from OpenCv = RGB
# photo[100:150, 200:280] = 119, 201, 105

# cv2.rectangle(photo, (0, 0), (100, 100), (119, 201, 105), thickness=3)
# cv2.line(photo, (150, 150), (300, 150), (119, 201, 105), thickness=3)
# cv2.circle(photo, (photo.shape[1] // 2, photo.shape[0] // 2), 50, (119, 201, 105), thickness=cv2.FILLED)

# cv2.putText(photo, 'TEXT', (300, 300), cv2.FONT_HERSHEY_TRIPLEX, 1, (119, 201, 105), 2)


# reflection_picture = cv2.flip(img, 0)

# Call functions from helpers folder
# img = rotate(img, 90)
# img = transform(img, 30, 200)


# new_image = np.zeros(img.shape, dtype='uint8')

# img = cv2.GaussianBlur(img, (5, 5), 0)
# img = cv2.Canny(img, 100, 140)
# con, hir = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# cv2.drawContours(new_image, con, -1, (230, 3, 40), 1)

# cv2.imshow('Photo', new_image)
# cv2.waitKey(0)


image = cv2.imread('images/people2.webp')
grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = cv2.CascadeClassifier('faces_neural_network.xml')

result = faces.detectMultiScale(grey_img, scaleFactor=1.1, minNeighbors=8)

for (x, y, w, h) in result:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)

cv2.imshow('Neural Network result', image)
cv2.waitKey(0)
