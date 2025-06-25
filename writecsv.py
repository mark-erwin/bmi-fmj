import csv
import cv2
import FaceLandmarks
import math
import pandas as pd

PARList = []
FWHRList = []
CJWRList = []

frontalface = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
n = 0

def detect_face(img):
    coord = frontalface.detectMultiScale(img, minNeighbors=7)
    for (x, y, w, h) in coord:
        area = w * h

        if area > 0:
            FaceLandmarks.detectBMI(x, y, w + 50, h + 50, img)
        else:
            break
    return img


def faceLandmarks(img):
    PAR, FWHR, CJWR = FaceLandmarks.trainBMI(img)
    PARList.append(PAR)
    FWHRList.append(FWHR)
    CJWRList.append(CJWR)
    return img


while True:
    if n <= 3962:
        print("---------------------------------------")
        filename = './Data/img_' + str(1) + '.bmp'
        image = cv2.imread(filename)
        print('img_' + str(n) + '.bmp')
        cv2.imshow('Preview', image)
        detect_face(image)
        n = n + 1
    #else:
    #    df = pd.read_csv("./data/trainingdata.csv")
    #    df['PAR'] = PARList
    #    df['FWHR'] = FWHRList
    #    df['CJWR'] = CJWRList
    #    df.to_csv("preprocessed.csv", index=False)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
