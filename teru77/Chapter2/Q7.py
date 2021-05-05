import cv2
import numpy as np

def Average_Poolimg(PATH: str,kernel_size: int):
    img = cv2.imread(PATH)
    dst = img.copy()
    
    w,h,c = img.shape
    #カーネルの中心画素から両端画素までの距離
    size = kernel_size // 2
    
    for x in range(size, w, kernel_size):
        for y in range(size, h, kernel_size):
            dst[x-size:x+size,y-size:y+size,0] = np.mean(img[x-size:x+size,y-size:y+size,0])
            dst[x-size:x+size,y-size:y+size,1] = np.mean(img[x-size:x+size,y-size:y+size,1])
            dst[x-size:x+size,y-size:y+size,2] = np.mean(img[x-size:x+size,y-size:y+size,2])

    cv2.imwrite('./img/Average_pooling_q7.jpg',dst)

if __name__ == '__main__':
    Average_Poolimg('./img/imori.jpg',8)

    