import cv2

def Gaussian_Filter(PATH: str):
    img = cv2.imread(PATH)
    output = cv2.GaussianBlur(img,(3,3),0)
    cv2.imwrite('./img/Gaussyan_Filter_q9.jpg',output)

if __name__ == '__main__':
    Gaussian_Filter('./img/imori_noise.jpg')