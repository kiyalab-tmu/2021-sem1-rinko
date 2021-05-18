import cv2

def BGR2GRAY(img):
    b = img[:,:,0].copy()
    g = img[:,:,1].copy()
    r = img[:,:,2].copy()

    out = r*0.2126 + g*0.7152 + b*0.0722

    return out

img = cv2.imread('./sample.jpg')
output = BGR2GRAY(img)

cv2.imwrite('./q2_output.png', output)
