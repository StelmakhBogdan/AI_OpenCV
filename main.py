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


# photo = np.zeros((600, 450, 3), dtype='uint8')

# BGR from OpenCv = RGB
# photo[100:150, 200:280] = 119, 201, 105

# cv2.rectangle(photo, (0, 0), (100, 100), (119, 201, 105), thickness=3)
# cv2.line(photo, (150, 150), (300, 150), (119, 201, 105), thickness=3)
# cv2.circle(photo, (photo.shape[1] // 2, photo.shape[0] // 2), 50, (119, 201, 105), thickness=cv2.FILLED)
#
# cv2.putText(photo, 'TEXT', (300, 300), cv2.FONT_HERSHEY_TRIPLEX, 1, (119, 201, 105), 2)
#
# cv2.imshow('Photo', photo)
# cv2.waitKey(0)
