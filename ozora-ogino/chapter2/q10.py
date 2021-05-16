import cv2
import numpy as np

Image = np.ndarray


def median_filter(img: Image, filter_size: int = 3) -> Image:
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


if __name__ == "__main__":
    img = cv2.imread("./img/imori_noise.jpg")
    cv2.imshow("Median filter", median_filter(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()