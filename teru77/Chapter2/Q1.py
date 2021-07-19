from PIL  import Image
import numpy as np
import cv2

input_img = './img/imori.jpg'

def Channel_Swapping():
    img = Image.open(input_img) # RGB
    img =np.asarray(img)
    cv2.imwrite('./img/BGR_q1.jpg',img) # BGR
    
if __name__ == '__main__':
    Channel_Swapping()
    