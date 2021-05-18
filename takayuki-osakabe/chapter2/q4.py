import cv2
import numpy as np

def otsu_binalization(img):
    hist = [np.sum(img == i) for i in range(256)]
    s_max = (0, -1*10**3)

    for th in range(256):
        p1 = sum(hist[:th])
        p2 = sum(hist[th:])

        r1 = p1 / img.shape[0]*img.shape[1]
        r2 = p2 / img.shape[0]*img.shape[1]

        if p1 == 0: 
            m1 = 0
        else: 
            m1 = sum([hist[i]*i for i in range(0,th)]) / p1
        
        if p2 == 0: 
            m2 = 0
        else: 
            m2 = sum([hist[i]*i for i in range(th,256)]) / p2

        s = r1 * r2 * (m1 - m2)**2

        if s > s_max[1]:
            s_max = (th, s)

    img[img < s_max[0]] = 0
    img[img >= s_max[0]] = 255

    return s_max[0], img


img = cv2.imread('./sample.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

th, output = otsu_binalization(img)
print(th)

cv2.imwrite('./q4_output.png', output)
