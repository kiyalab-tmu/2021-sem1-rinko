import cv2
import numpy as np

def Grayscale(PATH: str()):
    img = cv2.imread(PATH)  # uint8
    b,g,r = img[ : , : ,0],img[ : , : ,1],img[ : , : ,2]
    
    img_gray = 0.2126*r + 0.7152*g + 0.0722*b   # float64
    cv2.imwrite('./img/Gray_img_q2.jpg',img_gray.astype(np.uint8))  # uint8

if __name__ == '__main__':
    Grayscale('./img/imori.jpg')
