import cv2

def Median_Filter(PATH: str):
    img = cv2.imread(PATH)
    output = cv2.medianBlur(img,3)
    cv2.imwrite('./img/Median_Filter_q10.jpg',output)

if __name__ == '__main__':
    Median_Filter('./img/imori_noise.jpg')