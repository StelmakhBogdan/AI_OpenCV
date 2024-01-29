import cv2


def rotate(image, angle):
    height, width = image.shape[:2]
    point = (width // 2, height // 2)

    mat = cv2.getRotationMatrix2D(point, angle, 1)
    return cv2.warpAffine(image, mat, (width, height))
