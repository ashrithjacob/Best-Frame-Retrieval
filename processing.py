import torch
import matplotlib.pyplot as plt
import numpy as np
import os

get_epoch = lambda x: int(x.split("-")[2].split(".")[0][0:])


def get_latest_epoch(path):
    files = os.listdir(path)
    epochs = [
        get_epoch(x)
        for x in files
        if (x.endswith(".pth") and x.startswith("MS_SSIM_L1"))
    ]
    return max(epochs)


def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)


def resume(model, filename):
    model.load_state_dict(torch.load(filename))


def normalise_zero_one(*args):
    norm_img = []
    for x in args:
        x = (x * 0.5) + 0.5
        norm_img.append(x)
    return norm_img


def grey_to_rgb(img):
    if img.shape[0] == 1:
        torch.unsqueeze(img, 0)
        img = img.repeat(3, 1, 1)
    return img


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    img_t = np.transpose(img, (1, 2, 0))
    plt.imshow(img_t, cmap="hot")
    plt.show()


def imexpl(img):
    print(f"image shape: {img.shape}")
    print(f"image type: {img.dtype}")
    print(f"image min: {img.min()}")
    print(f"image max: {img.max()}")


x=get_latest_epoch("./checkpoints")
print(x)