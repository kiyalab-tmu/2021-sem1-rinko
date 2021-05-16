import cv2

def Discretization_of_Color(PATH: str):
    img = cv2.imread(PATH)

    img[(0<=img) & (img<63)] = 32
    img[(63<=img) & (img<127)] = 96
    img[(127<=img) & (img<191)] = 160
    img[(191<=img) & (img<256)] = 224
    
    cv2.imwrite('./img/Discretization_q6.jpg',img)

if __name__ == '__main__':
    Discretization_of_Color('./img/imori.jpg')