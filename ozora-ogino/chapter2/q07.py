import cv2
import numpy as np

Image = np.ndarray


def average_pool(img: Image, kernel_size: int = 8) -> Image:
    out = img.copy()
    for h in range(int(img.shape[0] / kernel_size)):
        for w in range(int(img.shape[1] / kernel_size)):
            for ch in range(img.shape[2]):
                out[
                    h * kernel_size : (h + 1) * kernel_size,
                    w * kernel_size : (w + 1) * kernel_size,
                    ch,
                ] = np.mean(
                    img[
                        h * kernel_size : (h + 1) * kernel_size,
                        w * kernel_size : (w + 1) * kernel_size,
                        ch,
                    ]
                )
    return out


if __name__ == "__main__":
    img = cv2.imread("./img/imori.jpg")
    cv2.imshow("Mean pooling", average_pool(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()