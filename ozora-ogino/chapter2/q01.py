import cv2
import numpy as np

Image = np.ndarray


def swap_ch(img: Image) -> Image:
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b

    return img


if __name__ == "__main__":
    img = cv2.imread("./img/imori.jpg")
    img = swap_ch(img)
    cv2.imshow("Swapped image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()