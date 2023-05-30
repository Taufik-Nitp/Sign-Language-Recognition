import numpy as np
import cv2
minValue = 70
# import mat
def func(path):    
    frame = cv2.imread(path)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)

    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    ret= cv2.resize(frame,(128,128));
    return res

res=func("/home/taufik/Desktop/SignLanguage/Sign-Language-to-Text/data/train/B/0.jpg")
cv2.imwrite("/home/taufik/Desktop/",res)

