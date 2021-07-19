import cv2
import numpy as np
import matplotlib.pyplot as plt

def HSV_Conversion(PATH: str):
    img = cv2.imread(PATH) # BGR
    img_hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV) # RGB→HSV
    img_hsv[:,:,0] = (img_hsv[:,:,0]+180)%180 # Hue of opencv is defined [0, 180]
    img_rgb = cv2.cvtColor(img_hsv,cv2.COLOR_HSV2RGB) # HSV→RGB
    cv2.imwrite('./img/HSV_Conversion_q5.jpg',img_rgb[:, :, ::-1])# BGRで書き出されないように
    
if __name__ == '__main__':
    HSV_Conversion('./img/imori.jpg')
    