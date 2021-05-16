import cv2
import numpy as np

Image = np.ndarray


def gray_scale(img: Image) -> Image:
    b, g, r = img[:, :, 0].copy(), img[:, :, 1].copy(), img[:, :, 2].copy()
    out = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return out.astype(np.uint8)


if __name__ == "__main__":
    img = cv2.imread("./img/imori.jpg")
    cv2.imshow("Gray scale", gray_scale(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()