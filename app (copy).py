from cgitb import reset
from tkinter import *
from tkinter.ttk import *
import tkinter as tk
import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import cvzone
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
from PIL import Image, ImageTk
import mediapipe as mp
import matplotlib.pyplot as plt

def capture():
    
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    classifier = Classifier("model-ALL/keras_model.h5", "model-ALL/labels.txt")

    offset = 20
    imgSize = 300

    folder = "Data/C"
    counter = 0
    s = ' '
    labels = ["A", "B", "C","D","E","G","H","I","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y"]

    while True:
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            # gray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)

            # blur = cv2.GaussianBlur(gray, (5, 5), 2)
            # th3 = cv2.adaptiveThreshold(blur, 255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            # ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
            imgCropShape = imgCrop.shape
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                imgStack = bg_remove(imgWhite)
                prediction, index = classifier.getPrediction(imgStack, draw=False)
                print(prediction, index)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                imgStack = bg_remove(imgWhite)

                prediction, index = classifier.getPrediction(imgStack, draw=False)

            cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                        (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.putText(imgOutput, str(prediction[index]*100), (x+60, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

            cv2.rectangle(imgOutput, (x-offset, y-offset),
                        (x + w+offset, y + h+offset), (255, 0, 255), 4)

            # if prediction[index]*100 >90:
            #     s += labels[index]
            #     print(s)
            # cv2.imshow("ImageCrop", imgCrop)
            # cv2.imshow("ImageWhite", imgWhite)

            cv2.imshow("Image", imgOutput)

            cv2.imshow("image", imgStack)
        # cv2.waitKey(1)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        # root.mainloop()

def bg_remove(img):
    segmentor = SelfiSegmentation()
    fpsReader = cvzone.FPS()
    imgOut = segmentor.removeBG(img, (0,0,0), threshold=0.83)
    # imgOut = segmentor.removeBG(img, imgList[indexImg], threshold=0.8)

    imgStack = cvzone.stackImages([img, imgOut], 2,1)
    _, imgStack = fpsReader.update(imgStack)
    return imgStack
    # key = cv2.waitKey(1)
    # if key == ord('q'):
        

def mask(img):
    change_background_mp = mp.solutions.selfie_segmentation
    change_bg_segment = change_background_mp.SelfieSegmentation()
    bg_img = cv2.imread('6.jpg')
    RGB_sample_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = change_bg_segment.process(RGB_sample_img)  
    binary_mask = result.segmentation_mask > 0.9
    binary_mask_3 = np.dstack((binary_mask,binary_mask,binary_mask))

    # output_image = np.where(binary_mask_3, sample_img, bg_img)     

    plt.figure(figsize=[22,22])
    plt.subplot(131)
    cv2.imshow('img', img[:,:,::-1]);plt.title("Original Image")
    plt.axis('off')
    plt.subplot(132)
    cv2.imshow('bin',binary_mask);plt.title("Binary Mask")
    plt.axis('off')
    # plt.subplot(133);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');

capture()

    

# root = tk.Tk()
# root.title("Sign Language To Text Conversion")
# # root.protocol('WM_DELETE_WINDOW',  destructor)
# root.geometry("900x650")
# panel = tk.Label( root)
# panel.place(x = 100, y = 10, width = 580, height = 580)
# panel2 = tk.Label( root) # initialize image panel
# panel2.place(x = 400, y = 65, width = 275, height = 275)
# T = tk.Label( root)
# T.place(x = 60, y = 5)
# T.config(text = "Sign Language To Text Conversion", font = ("Courier", 30, "bold"))
# panel3 = tk.Label( root) # Current Symbol
# panel3.place(x = 500, y = 540)
# T1 = tk.Label( root)
# T1.place(x = 10, y = 540)
# T1.config(text = "Character :", font = ("Courier", 15, "bold"))
# panel4 = tk.Label( root) # Word
# panel4.place(x = 220, y = 595)
# T2 = tk.Label( root)
# T2.place(x = 10,y = 595)
# T2.config(text = "Word :", font = ("Courier", 30, "bold"))
# panel5 = tk.Label( root) # Sentence
# panel5.place(x = 350, y = 645)
# T3 = tk.Label( root)
# T3.place(x = 10, y = 645)
# T3.config(text = "Sentence :",font = ("Courier", 30, "bold"))
# T4 = tk.Label( root)
# T4.place(x = 250, y = 690)
# T4.config(text = "Suggestions :", fg = "red", font = ("Courier", 30, "bold"))
# # bt1 = tk.Button( root, command =  action1, height = 0, width = 0)
# # bt1.place(x = 26, y = 745)
# # bt2 = tk.Button( root, command =  action2, height = 0, width = 0)
# # bt2.place(x = 325, y = 745)
# # bt3 = tk.Button( root, command =  action3, height = 0, width = 0)
# # bt3.place(x = 625, y = 745)
# # root.mainloop()
# capture()
            
    # offset = 20
    # imgSize = 300

    # folder = "Data/F"
    # counter = 0

    # while True:
    #     success, img = cap.read()
    #     hands, img = detector.findHands(img)
    #     if hands:
    #         hand = hands[0]
    #         x, y, w, h = hand['bbox']

    #         imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
    #         imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

    #         imgCropShape = imgCrop.shape

    #         gray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)

    #         blur = cv2.GaussianBlur(gray, (5, 5), 2)
    #         th3 = cv2.adaptiveThreshold(blur, 255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    #         ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #         imgtk = ImageTk.PhotoImage(image = res)   
    #         label.imgtk = imgtk
    #         label.configure(image = imgtk)
    #         aspectRatio = h / w

    #         if aspectRatio > 1:
    #             k = imgSize / h
    #             wCal = math.ceil(k * w)
    #             imgResize = cv2.resize(imgCrop, (wCal, imgSize))
    #             imgResizeShape = imgResize.shape
    #             wGap = math.ceil((imgSize - wCal) / 2)
    #             imgWhite[:, wGap:wCal + wGap] = imgResize

    #         else:
    #             k = imgSize / w
    #             hCal = math.ceil(k * h)
    #             imgResize = cv2.resize(imgCrop, (imgSize, hCal))
    #             imgResizeShape = imgResize.shape
    #             hGap = math.ceil((imgSize - hCal) / 2)
    #             imgWhite[hGap:hCal + hGap, :] = imgResize

    #         cv2.imshow("ImageCrop", res)
    #         # cv2.imshow("ImageWhite", imgWhite)

    #     cv2.imshow("Image", img)
    #     key = cv2.waitKey(1)
    #     if key == ord("s"):
    #         counter += 1
    #         cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
    #         print(counter)

