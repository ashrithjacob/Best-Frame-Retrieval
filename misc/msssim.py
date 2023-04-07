import torch
import numpy as np
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from PIL import Image, ImageFilter

def pil2tensor(im):  # in: [PIL Image with 3 channels]. out: [B=1, C=3, H, W] (0, 1)
    return torch.Tensor((np.float32(im) / 255).transpose(2, 0 ,1)).unsqueeze(0)

im1 = Image.open('images/frame1.png')
im2 = Image.open('images/frame2.png')
X = pil2tensor(im1)
Y = pil2tensor(im2)
# X: (N,3,H,W) a batch of non-negative RGB images (0~255)
# Y: (N,3,H,W)
print(X.shape)
print(torch.max(X))
print(torch.min(X))
print(Y.shape)
print(torch.max(Y))
print(torch.min(Y))
# calculate ssim & ms-ssim for each image
ssim_val = ssim(X, Y, data_range=1, size_average=False)  # return (N,)
ms_ssim_val = ms_ssim(X, Y, data_range=1, size_average=False)  # (N,)


print(ssim_val, ms_ssim_val)

# set 'size_average=True' to get a scalar value as loss. see tests/tests_loss.py for more details
ssim_loss = 1 - ssim(X, Y, data_range=1, size_average=True)  # return a scalar
ms_ssim_loss = 1 - ms_ssim(X, Y, data_range=1, size_average=True)
print(ssim_loss, ms_ssim_loss)

# reuse the gaussian kernel with SSIM & MS_SSIM.
ssim_module = SSIM(
    data_range=1, size_average=True, channel=3
)  # channel=1 for grayscale images
ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=3)

ssim_loss = 1 - ssim_module(X, Y)
ms_ssim_loss = 1 - ms_ssim_module(X, Y)
print(ssim_loss, ms_ssim_loss)