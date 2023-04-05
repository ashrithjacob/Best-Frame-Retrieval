import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

def grey_to_rgb(img):
    if img.shape[0] == 1:
        torch.unsqueeze(img, 0)
        img = img.repeat(3, 1, 1)
    return img

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    img_t = np.transpose(img, (1, 2, 0))
    plt.imshow(img_t, cmap='hot')
    plt.show()

def imexpl(img):
    print(f'image shape: {img.shape}')
    print(f'image type: {img.dtype}')
    print(f'image min: {img.min()}')
    print(f'image max: {img.max()}')


