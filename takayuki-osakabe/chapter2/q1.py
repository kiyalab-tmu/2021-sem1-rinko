import cv2

def RGB2BGR(img):
    r = img[:,:,0].copy()
    g = img[:,:,1].copy()
    b = img[:,:,2].copy()

    img[:,:,0] = b
    img[:,:,1] = g
    img[:,:,2] = r

    return img

if __name__ == '__main__':
    img = cv2.imread('./sample.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    output = RGB2BGR(img)
    
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    cv2.imwrite('./q1_output.png', output)
