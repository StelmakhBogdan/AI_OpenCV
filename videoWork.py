import cv2

capture = cv2.VideoCapture('video/car_video.mp4')
# capture = cv2.VideoCapture(0)
# if i want to track video from leave camera just set cv2.VideoCapture(0) or 1,2,3 number of your camera
capture.set(3, 600)
capture.set(4, 400)

while True:
    success, img = capture.read()
    cv2.imshow('Video Result', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
