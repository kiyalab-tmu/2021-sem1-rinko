import cv2
import numpy as np

Image = np.ndarray


def rgb2hsv(rgb: Image) -> Image:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


def hsv2rgb(hsv: Image) -> Image:
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


if __name__ == "__main__":
    img = cv2.imread("./img/imori.jpg")
    hsv = rgb2hsv(img)
    hsv[:, :, 0] = (hsv[:, :, 0] + 180) % 360
    cv2.imshow("HSV Conversion", hsv2rgb(hsv))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
