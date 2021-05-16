import cv2
import numpy as np

def Max_Poolimg(PATH: str,kernel_size: int):
    img = cv2.imread(PATH)
    output = img.copy()
    
    w,h,c = img.shape
    #カーネルの中心画素から両端画素までの距離
    size = kernel_size // 2
    
    for x in range(size, w, kernel_size):
        for y in range(size, h, kernel_size):
            output[x-size:x+size,y-size:y+size,0] = np.max(img[x-size:x+size,y-size:y+size,0])
            output[x-size:x+size,y-size:y+size,1] = np.max(img[x-size:x+size,y-size:y+size,1])
            output[x-size:x+size,y-size:y+size,2] = np.max(img[x-size:x+size,y-size:y+size,2])

    cv2.imwrite('./img/Max_pooling_q8.jpg',output)

if __name__ == '__main__':
    Max_Poolimg('./img/imori.jpg',8)