import cv2
import numpy as np
import time, argparse

cap = cv2.VideoCapture(0)

time.sleep(3)

for i in range(60):
    ret, background = cap.read()

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue 

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(img, lower_red, upper_red)

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(img, lower_red, upper_red)

    mask1 = mask1 + mask2

    cv2.imshow("img", img)

    if cv2.waitKey(1) == ord('q'):
        break
