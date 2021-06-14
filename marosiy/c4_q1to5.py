#!/usr/bin/env python
# coding: utf-8


from PIL import Image
import numpy as np

#データの読み込み
img = np.asarray(Image.open('c4_src/pic.jpg'))
img_gray = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]


#変数設定
karnel_size = 3
stride = 1
karnel = np.array(
    [[1, 0, -1],
    [0, 0, 0],
    [-1, 0, 1]])
is_padding = True


#paddingの実行
if is_padding:
    img_gray = np.pad(img_gray, [(1,1),(1,1)] ,'constant')

#畳み込みの実行
new_img = list()
for i in range((img_gray.shape[0] - karnel_size) //stride):
    new_img_retu = list()
    for j in range((img_gray.shape[1] - karnel_size) //stride):
        new_pixcel = np.sum(img_gray[i:i+karnel_size, j:j+karnel_size] * karnel) #畳みこみ
        # new_pixcel = np.max(img_gray[i:i+karnel_size, j:j+karnel_size] * karnel) #プーリング
        new_img_retu.append(new_pixcel)
    new_img.append(new_img_retu)
new_img = np.array(new_img)
print(new_img.shape)


#表示
generated_img = Image.fromarray(new_img)
generated_img.show()


