import cv2
import numpy as np

Image = np.ndarray


def dicrease_color(img: Image) -> Image:
    img[(img >= 0) == (img < 63)] = 32
    img[(img >= 63) == (img < 127)] = 96
    img[(img >= 127) == (img < 191)] = 160
    img[(img >= 191) == (img < 224)] = 224
    return img


if __name__ == "__main__":
    img = cv2.imread("./img/imori.jpg")
    cv2.imshow("Dicretization of color", dicrease_color(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()