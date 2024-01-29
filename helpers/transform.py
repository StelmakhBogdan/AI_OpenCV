import cv2
import numpy as np


def transform(image, x, y):
    height, width = image.shape[:2]
    mat = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(image, mat, (width, height))