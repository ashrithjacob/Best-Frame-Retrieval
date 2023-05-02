import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from model import Autoencoder
from torchvision.transforms import ToTensor


class Helper:
    get_epoch = lambda x: int(x.split("-")[2].split(".")[0][0:])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def load_model():
        """
        Load the model for inference.
        """
        model = Autoencoder().to(Helper.device)
        checkpoint_dir = "./checkpoints"
        loading_epoch = Helper.get_latest_epoch(checkpoint_dir)
        if loading_epoch != 0:
            Helper.resume(model, f"{str(checkpoint_dir)}/MS_SSIM_L1-epoch-{str(loading_epoch)}.pth")
            print(f"start inference from epoch {str(loading_epoch)}")
        else:
            raise Exception("No checkpoint found, please upload checkpoint for inference")
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
