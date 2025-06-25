import csv
import cv2
import face_detection
import math
import pandas as pd
import dlib


def write_landmarks(img_count):
    img_names = []
    par_list = []
    fwhr_list = []
    cjwr_list = []

    n = 0
    while n <= img_count:
        filename = r'E:\GitHub\face-to-bmi-vit\data\Images\img_' + str(n) + '.bmp'
        image = cv2.imread(filename)

        if image is None:  # Skip if image does not exist
            n = n + 1
            continue

        name = 'img_' + str(n) + '.bmp'
        PAR, FWHR, CJWR = face_detection.detect_landmarks(image)

        if PAR == 0 or FWHR == 0 or CJWR == 0:  # Skip if image does not exist
            n = n + 1
            continue

        img_names.append(name)
        par_list.append(PAR)
        fwhr_list.append(FWHR)
        cjwr_list.append(CJWR)

        df = pd.DataFrame({
            "Image": img_names,
            "PAR": par_list,
            "FWHR": fwhr_list,
            "CJWR": cjwr_list
        })

        print("img_" + str(n) + ": " + str(PAR) + ", " + str(FWHR) + ", " + str(CJWR))
        n = n + 1

    df.to_csv("facelandmarks.csv", index=False)


def merge_bmi_landmarks(bmi_df, lmark_df):
    merged_df = pd.merge(bmi_df, lmark_df, on='image', how='inner')
    merged_df.to_csv("bmi-landmarks.csv", index=False)


# write_landmarks(4205)

bmi_csv = pd.read_csv("bmi.csv")  # Main file
lmark_csv = pd.read_csv("facelandmarks.csv")
merge_bmi_landmarks(bmi_csv, lmark_csv)
