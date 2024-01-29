import cv2
import numpy as np

capture = cv2.VideoCapture('video/car_video.mp4')
# capture = cv2.VideoCapture(0)
# if i want to track video from leave camera just set cv2.VideoCapture(0) or 1,2,3 number of your camera
capture.set(3, 600)
capture.set(4, 400)

while True:
    success, img = capture.read()

    bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    new_img = cv2.Canny(bw_img, 30, 30)

    kernel = np.ones((5, 5), np.uint8)

    dilate = cv2.dilate(new_img, kernel, iterations=1)
    erode = cv2.erode(dilate, kernel, iterations=1)

    cv2.imshow('Video Result', erode)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


