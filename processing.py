import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

def normalise_zero_one(*args): 
    for x in args:
        (x - x.min()) / (x.max() - x.min())

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

def ms_ssim_loss(img1, img2):
    normalise_zero_one(img1, img2)
    ms_ssim = MS_SSIM(data_range=1, size_average=True, channel=3)
    return (1 - ms_ssim(img1, img2))


