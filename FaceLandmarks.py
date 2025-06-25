import cv2
import dlib
import math
import pandas as pd
from joblib import load
import os

# Load the model
model = load('./models/catboost_model.pkl')
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
BMIClasses = ["Severely Underweight", "Underweight", "Normal Weight", "Overweight", "Obese"]


def trainBMI(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = hog_face_detector(gray)
    perimeterLowerXY = []
    lowerPerimeter = 0
    lowerArea = 0
    PAR = 0
    FWHR = 0
    CJWR = 0

    for face in faces:
        perimeterLowerXY = []
        face_landmarks = dlib_facelandmark(gray, face)
        for n in range(0, 68):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            cv2.circle(img, (x, y), 1, (0, 255, 255), 1)

        # Points 1 - 17
        for n in range(0, 17):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            coords = (x, y)
            perimeterLowerXY.append(coords)

        x1 = face_landmarks.part(0).x
        y1 = face_landmarks.part(0).y
        coords1 = (x1, y1)
        perimeterLowerXY.append(coords1)

        # Add
        for i in range(len(perimeterLowerXY) - 1):
            coord1 = perimeterLowerXY[i]
            coord2 = perimeterLowerXY[i + 1]

            x2 = coord2[0]
            x1 = coord1[0]
            y2 = coord2[1]
            y1 = coord1[1]

            cv2.line(img, (x2, y2), (x1, y1), (255, 0, 0), 1)
            pointsDistance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            lowerPerimeter = lowerPerimeter + pointsDistance

        # Area
        area = 0
        for i in range(len(perimeterLowerXY)):
            j = (i + 1) % len(perimeterLowerXY)
            area += perimeterLowerXY[i][0] * perimeterLowerXY[j][1]
            area -= perimeterLowerXY[j][0] * perimeterLowerXY[i][1]
        lowerArea = abs(area) / 2.0

        PAR = lowerPerimeter / lowerArea

        # lcheekbone
        xlcheekbone = face_landmarks.part(0).x
        ylcheekbone = face_landmarks.part(0).y
        coordslcheekbone = (xlcheekbone, ylcheekbone)

        # rcheekbone
        xrcheekbone = face_landmarks.part(16).x
        yrcheekbone = face_landmarks.part(16).y
        coordsrcheekbone = (xrcheekbone, yrcheekbone)

        cv2.line(img, (xrcheekbone, yrcheekbone), (xlcheekbone, ylcheekbone), (0, 255, 0), 1)

        faceWidth = math.sqrt((xrcheekbone - xlcheekbone) ** 2 + (yrcheekbone - ylcheekbone) ** 2)

        # upperLipSuperior
        xupperLipSuperior = face_landmarks.part(51).x
        yupperLipSuperior = face_landmarks.part(51).y
        coordsupperLipSuperior = (xupperLipSuperior, yupperLipSuperior)

        # eyeBrowInferior
        xeyeBrowInferior = face_landmarks.part(27).x
        yeyeBrowInferior = face_landmarks.part(27).y
        coordseyeBrowInferior = (xeyeBrowInferior, yeyeBrowInferior)

        faceHeight = math.sqrt(
            (xeyeBrowInferior - xupperLipSuperior) ** 2 + (yeyeBrowInferior - yupperLipSuperior) ** 2)

        cv2.line(img, (xeyeBrowInferior, yeyeBrowInferior), (xupperLipSuperior, yupperLipSuperior), (0, 255, 0), 1)

        FWHR = faceWidth / faceHeight

        # lJaw
        xlJaw = face_landmarks.part(4).x
        ylJaw = face_landmarks.part(4).y
        coordslJaw = (xlJaw, ylJaw)

        # rRaw
        xrJaw = face_landmarks.part(12).x
        yrJaw = face_landmarks.part(12).y
        coordsrJaw = (xrJaw, yrJaw)

        cv2.line(img, (xlJaw, ylJaw), (xrJaw, yrJaw), (0, 0, 255), 1)

        jawDistance = math.sqrt((xrJaw - xlJaw) ** 2 + (yrJaw - ylJaw) ** 2)

        CJWR = faceWidth / jawDistance

        print("PerimeterAreaR: " + str(PAR))
        print("FaceWHR: " + str(FWHR))
        print("CheekJawWR: " + str(CJWR))

    cv2.imshow("Face Landmarks", img)
    return PAR, FWHR, CJWR


def detectBMI(x, y, w, h, img):
    cropped = img[y:y + h, x:x + w]
    resized = cv2.resize(cropped, (224, 224))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    faces = hog_face_detector(gray)
    perimeterLowerXY = []
    lowerPerimeter = 0
    lowerArea = 0
    PAR = 0
    FWHR = 0
    CJWR = 0

    for face in faces:
        perimeterLowerXY = []
        face_landmarks = dlib_facelandmark(gray, face)
        for n in range(0, 68):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            cv2.circle(resized, (x, y), 1, (0, 255, 255), 1)

        # Points 1 - 17
        for n in range(0, 17):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            coords = (x, y)
            perimeterLowerXY.append(coords)

        x1 = face_landmarks.part(0).x
        y1 = face_landmarks.part(0).y
        coords1 = (x1, y1)
        perimeterLowerXY.append(coords1)

        # Add
        for i in range(len(perimeterLowerXY) - 1):
            coord1 = perimeterLowerXY[i]
            coord2 = perimeterLowerXY[i + 1]

            x2 = coord2[0]
            x1 = coord1[0]
            y2 = coord2[1]
            y1 = coord1[1]

            cv2.line(resized, (x2, y2), (x1, y1), (255, 0, 0), 1)
            pointsDistance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            lowerPerimeter = lowerPerimeter + pointsDistance

        # Area
        area = 0
        for i in range(len(perimeterLowerXY)):
            j = (i + 1) % len(perimeterLowerXY)
            area += perimeterLowerXY[i][0] * perimeterLowerXY[j][1]
            area -= perimeterLowerXY[j][0] * perimeterLowerXY[i][1]
        lowerArea = abs(area) / 2.0

        PAR = lowerPerimeter / lowerArea

        # lcheekbone
        xlcheekbone = face_landmarks.part(0).x
        ylcheekbone = face_landmarks.part(0).y
        coordslcheekbone = (xlcheekbone, ylcheekbone)

        # rcheekbone
        xrcheekbone = face_landmarks.part(16).x
        yrcheekbone = face_landmarks.part(16).y
        coordsrcheekbone = (xrcheekbone, yrcheekbone)

        cv2.line(resized, (xrcheekbone, yrcheekbone), (xlcheekbone, ylcheekbone), (0, 255, 0), 1)

        faceWidth = math.sqrt((xrcheekbone - xlcheekbone) ** 2 + (yrcheekbone - ylcheekbone) ** 2)

        # upperLipSuperior
        xupperLipSuperior = face_landmarks.part(51).x
        yupperLipSuperior = face_landmarks.part(51).y
        coordsupperLipSuperior = (xupperLipSuperior, yupperLipSuperior)

        # eyeBrowInferior
        xeyeBrowInferior = face_landmarks.part(27).x
        yeyeBrowInferior = face_landmarks.part(27).y
        coordseyeBrowInferior = (xeyeBrowInferior, yeyeBrowInferior)

        faceHeight = math.sqrt(
            (xeyeBrowInferior - xupperLipSuperior) ** 2 + (yeyeBrowInferior - yupperLipSuperior) ** 2)

        cv2.line(resized, (xeyeBrowInferior, yeyeBrowInferior), (xupperLipSuperior, yupperLipSuperior), (0, 255, 0), 1)

        FWHR = faceWidth / faceHeight

        # lJaw
        xlJaw = face_landmarks.part(4).x
        ylJaw = face_landmarks.part(4).y
        coordslJaw = (xlJaw, ylJaw)

        # rRaw
        xrJaw = face_landmarks.part(12).x
        yrJaw = face_landmarks.part(12).y
        coordsrJaw = (xrJaw, yrJaw)

        cv2.line(resized, (xlJaw, ylJaw), (xrJaw, yrJaw), (0, 0, 255), 1)

        jawDistance = math.sqrt((xrJaw - xlJaw) ** 2 + (yrJaw - ylJaw) ** 2)

        CJWR = faceWidth / jawDistance

        print("PerimeterAreaR: " + str(PAR))
        print("FaceWHR: " + str(FWHR))
        print("CheekJawWR: " + str(CJWR))

        data = pd.DataFrame({
            'PAR': [PAR],
            'FWHR': [FWHR],
            'CJWR': [CJWR]
        }, columns=['PAR', 'FWHR', 'CJWR'])

        predicted_bmi = round(model.predict(data)[0] - 6.305612453733344, 1)

        if predicted_bmi <= 16:
            bmiclass = BMIClasses[0]
        elif 16 < predicted_bmi <= 18.5:
            bmiclass = BMIClasses[1]
        elif 18.5 < predicted_bmi <= 24.9:
            bmiclass = BMIClasses[2]
        elif 25 < predicted_bmi <= 29.9:
            bmiclass = BMIClasses[3]
        elif 30 <= predicted_bmi:
            bmiclass = BMIClasses[4]
        else:
            print("Error: Predicted BMI is out of range")  # Add this line to check if none of the conditions match
            bmiclass = "Unknown"

        resized = cv2.resize(resized, (448, 448))

        textBMI = "Predicted BMI: " + str(predicted_bmi) + " Class: " + bmiclass

        print(textBMI)
        resized = cv2.putText(resized, textBMI, (20, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness=1, lineType=cv2.LINE_AA, color=(255, 255, 255))

    length = len(os.listdir("./captures"))
    cv2.imshow("Face Landmarks", resized)
    cv2.imwrite("./captures/capture_" + str(length) + ".png", resized)
