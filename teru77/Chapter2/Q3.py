import cv2

def Binarization(PATH: str):
    img = cv2.imread(PATH,0)    # input grayscale image
    _,img_th = cv2.threshold(img,128,255,cv2.THRESH_BINARY)
    cv2.imwrite('./img/Binarization_q3.jpg',img_th)

if __name__ == '__main__':
    Binarization('./img/imori.jpg')