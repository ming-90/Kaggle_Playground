from operator import delitem
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands = 1,
    min_detection_confidence = 0.5,
    min_tracking_confidence= 0.5)

file = np.genfromtxt('gesture_train.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    #img = cv2.flip(img, 1)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    if result.multi_hand_landmarks is not None:
        print(result.multi_hand_landmarks)

    cv2.imshow('asdf', img)
    if cv2.waitKey(1) == ord('q'):
        break

