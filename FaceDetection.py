import cv2
import csv
import FaceLandmarks
import os

cam = cv2.VideoCapture(1)

frontalface = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if not os.path.exists("./captures"):
    os.mkdir("./captures")


def detect_face(img):
    coord = frontalface.detectMultiScale(img, minNeighbors=7)
    for (x,y,w,h) in coord:
        area = w*h
        
        if area > 0:
            FaceLandmarks.detectBMI(x,y,w+50,h+50,img)
        else:
            break
    return img


while True:
    ret,image = cam.read()
    cv2.imshow('Preview',image)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        image = detect_face(image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

                