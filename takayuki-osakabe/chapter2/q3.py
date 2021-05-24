import cv2

def binalization(img, th=128):
    b = img[:,:,0].copy()
    g = img[:,:,1].copy()
    r = img[:,:,2].copy()

    gray = 0.2126*r + 0.7152*g + 0.0722*b

    gray[gray < th] = 0
    gray[gray >= th] = 255

    return gray

img = cv2.imread('./sample.jpg')

output = binalization(img)

cv2.imwrite('./q3_output.png', output)
