import cv2
import numpy as np
from scipy import signal

def gaussian(img, filter_size=3, sigma=1.3):
    output = np.empty((img.shape))
    
    gaussian = np.zeros((filter_size, filter_size), np.float32)
    m = filter_size//2
    n = filter_size//2

    for x in range(-m, m+1):
        for y in range(-n, n+1):
            x1 = 2*np.pi*(sigma**2)
            x2 = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            gaussian[x+m, y+n] = (1/x1)*x2
    print(gaussian)

    pad_img = np.pad(img, (1,1), 'edge')

    for c in range(img.shape[2]):
        output[:,:,c] = signal.convolve2d(pad_img[:,:,c], gaussian, 'valid')

    return output

img = cv2.imread("./sample_noise.jpeg")

output = gaussian(img)

cv2.imwrite('./q9_output.png', output)
