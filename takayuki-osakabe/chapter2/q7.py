import numpy as np
import cv2

def im2col(img, fh, fw, stride=1, pad=0):
    N, C, H, W = img.shape
    out_h = (H + 2*pad - fh)//stride + 1
    out_w = (W + 2*pad - fw)//stride + 1

    img = np.pad(img, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')
    col = np.zeros((N, C, fh, fw, out_h, out_w))
    
    for y in range(fh):
        y_max = y + stride*out_h
        for x in range(fw):
            x_max = x + stride*out_w
            col[:,:,y,x,:,:] = img[:,:,y:y_max:stride,x:x_max:stride]

    col = col.transpose(0,4,5,1,2,3).reshape(N*out_h*out_w, -1)

    return col

def average_pooling(img, fh, fw, stride=1, pad=0):
    N, C, H, W  = img.shape
    out_h = int(1 + (H - fh) / stride)
    out_w = int(1 + (W - fw) / stride)

    col = im2col(img, fh, fw, stride, pad)
    col = col.reshape(-1, fh*fw)

    out = np.mean(col, axis=1)
    out = out.reshape(N, out_h, out_w, C).transpose(0,3,1,2)

    return out


img = cv2.imread('./sample.jpg')

out = average_pooling(img[np.newaxis,:,:,:].transpose(0,3,1,2), 8, 8, stride=8)
cv2.imwrite('./q7_output.png', out[0,:,:,:].transpose(1,2,0))
