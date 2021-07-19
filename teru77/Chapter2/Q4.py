import cv2

def Binarization_of_Otsu(PATH: str):
    img = cv2.imread(PATH,0)    #input grayscale image
    ret,img_th = cv2.threshold(img,0,255,cv2.THRESH_OTSU) #use Otsu algorithm
    print('閾値: {}'.format(ret))
    cv2.imwrite('./img/Binarization_Otsu_q4.jpg',img_th)

if __name__ == '__main__':
    Binarization_of_Otsu('./img/imori.jpg')