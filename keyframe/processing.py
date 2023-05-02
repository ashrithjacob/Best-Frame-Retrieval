import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import re
from torch import nn
from model import Autoencoder
from PIL import Image, ImageFilter
from torchvision.transforms import ToTensor


class Helper:
    get_epoch = lambda x: int(x.split("-")[2].split(".")[0][0:])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pat=re.compile("(\d+)\D*$")

    @staticmethod
    def get_latest_epoch(path):
        files = os.listdir(path)
        epochs = [
            Helper.get_epoch(x)
            for x in files
            if (x.endswith(".pth") and x.startswith("MS_SSIM_L1"))
        ]
        if len(epochs) == 0:
            return 0
        else:
            return max(epochs)

    @staticmethod
    def checkpoint(model, filename):
        torch.save(model.state_dict(), filename)

    @staticmethod
    def resume(model, filename):
        model.load_state_dict(torch.load(filename))

    @staticmethod
    def normalise_zero_one(*args):
        norm_img = []
        for x in args:
            x = (x * 0.5) + 0.5
            norm_img.append(x)
        return norm_img

    @staticmethod
    def grey_to_rgb(img):
        if img.shape[0] == 1:
            torch.unsqueeze(img, 0)
            img = img.repeat(3, 1, 1)
        return img

    @staticmethod
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        img_t = np.transpose(img, (1, 2, 0))
        plt.imshow(img_t, cmap="hot")
        plt.show()

    @staticmethod
    def display_pairs(img1, img2, title_1, title_2):
        """
        Takes two PIL images and displays them side by side.

        Parameters:
        img1 (PIL.Image): The first image.
        img2 (PIL.Image): The second image.
        """
        # Display ground truth image and blurred image side by side
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        # Display ground truth image
        ax[0].imshow(img1)
        ax[0].set_title(str(title_1))
        # Display blurred image
        ax[1].imshow(img2)
        ax[1].set_title(str(title_2))
        plt.show()

    @staticmethod
    def imexpl(img):
        print(f"image shape: {img.shape}")
        print(f"image type: {img.dtype}")
        print(f"image min: {img.min()}")
        print(f"image max: {img.max()}")

    @staticmethod
    def frame_rate(video):
        """
        Get the frame rate of a video.

        Parameters:
        video (cv2.VideoCapture): A cv2.VideoCapture object representing the video.

        Returns:
        float: The frame rate of the video.
        """
        return video.get(cv2.CAP_PROP_FPS)

    @staticmethod
    def bgr_to_rgb(image):
        """
        Convert a BGR image to RGB format.

        Parameters:
        image (numpy.ndarray): A 3-dimensional numpy array representing the BGR image.

        Returns:
        numpy.ndarray: A 3-dimensional numpy array representing the RGB image.
        """
        return image[..., ::-1]

    @staticmethod
    def load_model(use_model=True):
        """
        Load the model for inference.
        """
        if use_model:
            model = Autoencoder().to(Helper.device)
            checkpoint_dir = "./checkpoints"
            loading_epoch = Helper.get_latest_epoch(checkpoint_dir)
            if loading_epoch != 0:
                Helper.resume(model, f"{str(checkpoint_dir)}/MS_SSIM_L1-epoch-{str(loading_epoch)}.pth")
                print(f"start inference from epoch {str(loading_epoch)}")
            else:
                raise Exception("No checkpoint found, please upload checkpoint for inference")
        else:
            model = nn.Identity().to(Helper.device)
        model.eval()
        return model

    @staticmethod
    def transform(frame):
        frame = Helper.bgr_to_rgb(frame)
        frame = ToTensor()(frame.copy())
        frame = torch.unsqueeze(frame, axis=0)
        frame = frame.to(Helper.device)
        return frame

    @staticmethod
    def mkdir(directory):
        """
        Create a directory if it doesn't exist.

        Parameters:
        directory (str): The path to the directory to be created.
        """
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print("Error: Creating directory for frame storage:", str(directory))

    @staticmethod
    def key_func(x):
            mat=Helper.pat.search(os.path.split(x)[-1]) # match last group of digits
            if mat is None:
                return x
            return "{:>10}".format(mat.group(1)) # right align to 10 digits.

    @staticmethod
    def MS_SSIM_L1_diff(generator, function, model):
        diff = []
        counter = 0
        try:
            frame1 = next(generator)
            while True:
                frame2 = next(generator)
                diff.append(function(model(Helper.transform(frame1)), model(Helper.transform(frame2))).item())
                count += 1
                frame1 = frame2
        except StopIteration:
            print(f"loaded diffs: DONE")
        return diff, count
