import cv2
import numpy as np

def median_filter(img, filter_size=3):
    H, W, C = img.shape
    P = filter_size // 2

    out = np.zeros((H + P * 2, W + P * 2, C), dtype=np.float)
    out[P : P + H, P : P + W] = img.copy().astype(np.float)
    tmp = out.copy()

    # Filtering
    for h in range(H):
        for w in range(W):
            for ch in range(C):
                out[P + h, P + w, ch] = np.median(
                    tmp[h : h + filter_size, w : w + filter_size, ch]
                )
    return out[P : P + H, P : P + H].astype(np.uint8)

img = cv2.imread("./img/imori_noise.jpg")
