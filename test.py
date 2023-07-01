import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detect = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 310

folder = "Images/Z"
c = 0

lables = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
while True:
    success, img1 = cap.read()
    imgOutput = img1.copy()
    hands, img = detect.findHands(img1)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((310, 310, 3), np.uint8)*255
        imgCrop = img[y-offset:y + h+offset, x-offset:x + w+offset]

        imageCropShape = imgCrop.shape

        ratio = h/w
        if ratio > 1:
            k = 310/h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, 310))
            imageResizeShape = imgResize.shape
            wGap = math.ceil((310-wCal)/2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        else:
            k = 310 / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (310, hCal))
            imageResizeShape = imgResize.shape
            hGap = math.ceil((310 - hCal) / 2)
            imgWhite[hGap:hCal+hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),(x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, lables[index], (x,y-20), cv2.FONT_HERSHEY_COMPLEX, 2 , (255,0,255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x + w+offset, y + h+offset),(255,0,255),4)
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)

