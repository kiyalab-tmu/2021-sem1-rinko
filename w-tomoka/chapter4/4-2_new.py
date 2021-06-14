import numpy as np
import torch 
import torch.nn.functional as F
from matplotlib import pyplot as plt
from skimage import io
import cv2

from google.colab.patches import cv2_imshow

!curl -o img_ori https://yoyoyo-yo.github.io/Gasyori100knock/dataset/images/imori_256x256.png
img_ori = cv2.imread('img_ori', cv2.IMREAD_UNCHANGED)

#img_ori = io.imread('https://yoyoyo-yo.github.io/Gasyori100knock/dataset/images/imori_256x256.png')
#img_ori = io.imread('https://upload.wikimedia.org/wikipedia/commons/5/50/Vd-Orig.png')
img_gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
cv2_imshow(img_gray)
#img = np.array(img_gray, np.float32)[np.newaxis, np.newaxis]
img = torch.as_tensor(np.array(img_gray, np.float32))
gray_kernel = np.array([0.299, 0.587, 0.114], np.float32).reshape(1, 3, 1, 1)  # color -> gray kernel
gray_k = torch.as_tensor(gray_kernel)
gray = torch.sum(img * gray_k, dim=1, keepdim=True)  # grayscale image [1, 1, H, W]
gray_img = torch.as_tensor(gray)


#W = np.array([[1, 0, -1],[0, 0, 0],[-1,0,1]], np.float32)
#W = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], np.float32)
W = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], np.float32)
W = W[np.newaxis, np.newaxis]
W_kernel = torch.as_tensor(W)

result = F.conv2d(gray_img, W_kernel)


plt.figure(figsize=(12, 3))
plt.subplot(1, 2, 1)
plt.title('input')
plt.imshow(gray_img[0][0], cmap='gray')
plt.subplot(1, 2, 2)
plt.title('answer')
plt.imshow(result[0][0], cmap='gray')

plt.show()

result = result[0][0].numpy()




cv2_imshow(result)
