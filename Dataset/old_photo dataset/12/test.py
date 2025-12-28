import cv2
import numpy as np

img = cv2.imread("0010053x1.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, crack_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

kernel = np.ones((3, 3), np.uint8)
crack_mask = cv2.dilate(crack_mask, kernel, iterations=1)

cv2.imwrite("crack_mask.png", crack_mask)
