import cv2
import numpy as np

Image = np.ndarray


def gray_scale(img: Image) -> Image:
    b, g, r = img[:, :, 0].copy(), img[:, :, 1].copy(), img[:, :, 2].copy()
    out = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return out.astype(np.uint8)


def binarize(img: Image, thr: int = 128) -> Image:
    img[img < thr] = 0
    img[img >= thr] = 255
    return img


if __name__ == "__main__":
    img = cv2.imread("./img/imori.jpg")
    img = gray_scale(img)
    cv2.imshow("Binarized", binarize(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()