import cv2
import csv
import face_detection
import os

cam = cv2.VideoCapture(0)

if not os.path.exists("./captures"):
    os.mkdir("./captures")

while True:
    ret, image = cam.read()
    cv2.imshow('Preview', image)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        face_detection.predict_bmi(face_detection.face_dimensions(image),image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
