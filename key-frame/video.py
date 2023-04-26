import cv2
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import os
import sys
import time
from torchvision.transforms import ToTensor
from MS_SSIM_L1_loss import MS_SSIM_L1_LOSS


class Video:
    def __init__(self, path, per_second_acquisition=5, threshold=50):
        """
        class to extract frames from a video

        Parameters:
        path (str): The path to the video.
        per_second_acquisition (int): The number of frames to be extracted per second.

        Attributes:
        path (str): The path to the video.
        per_second_acquisition (int): The number of frames to be extracted per second.
        deblurred_frames_generator (generator): A generator object that yields the deblurred frames of the video.
        """
        self.path = path
        self.per_second_acquisition = per_second_acquisition
        self.threshold = threshold
        self.deblurred_frames_generator = None

    def get_generators(self):
        """
        Create object generators from the video.

        Attributes:
        frames_generator (generator): A generator object that yields the frames of the video.
        """
        frames_generator = self.extract(cv2.VideoCapture(self.path))
        self.deblurred_frames_generator = self.deblur(frames_generator)
        print(f"Created generator objects")
        frames_generator = None

    def frame_rate(self, video):
        """
        Get the frame rate of a video.

        Parameters:
        video (cv2.VideoCapture): A cv2.VideoCapture object representing the video.

        Returns:
        float: The frame rate of the video.
        """
        return video.get(cv2.CAP_PROP_FPS)

    def extract(self, video):
        """"
        Get the frames of a video at a certain rate('per_second_acquisition' frames per second)

        Parameters:
        video (cv2.VideoCapture): A cv2.VideoCapture object representing the video.
        per_second_acquisition (int): The number of frames to be extracted per second.

        Returns:
        generator: A generator object that yields the frames of the video.
        """
        current_frame = 0
        fps = self.frame_rate(video)
        print(f"FPS: {fps}")
        collect_every = fps // self.per_second_acquisition #Raiseerror if fps < per_second_acquisition
        while True:
            success, frame = video.read()
            if success:
                if current_frame % collect_every == 0:
                    yield frame
                current_frame += 1
            else:
                break
        video.release()
        cv2.destroyAllWindows()

    def bgr_to_rgb(self, image):
        """
        Convert a BGR image to RGB format.

        Parameters:
        image (numpy.ndarray): A 3-dimensional numpy array representing the BGR image.

        Returns:
        numpy.ndarray: A 3-dimensional numpy array representing the RGB image.
        """
        return image[..., ::-1]

    def deblur(self, frames_generator):
        max_laplacian_var = 0
        count = 0
        max_laplacian_var_frame = None
        for frame in frames_generator:
            if max_laplacian_var < cv2.Laplacian(frame, cv2.CV_64F).var():
                max_laplacian_var = cv2.Laplacian(frame, cv2.CV_64F).var()
                max_laplacian_var_frame = frame
            count += 1
            if count % self.per_second_acquisition == 0:
                yield max_laplacian_var_frame
                max_laplacian_var = 0
                count = 0

    def save(self,directory, type = 'blur'):
        """
        Save the frames of the video in a directory.

        Parameters:
        directory (str): The path to the directory where the frames are to be saved.
        """
        self.mkdir(directory)
        if self.deblurred_frames_generator is None:
            self.get_generators()
        if type == 'blur':
            self.save_deblurred(directory)
        elif type == 'key':
            self.save_key_frames(directory)
        self.reset()

    def save_deblurred(self, directory):
        for i, frame in enumerate(self.deblurred_frames_generator):
                cv2.imwrite(os.path.join(directory, f"s_{i}.png"), frame)

    def save_key_frames(self, directory):
        i = 1
        try:
            frame1 = self.transform(next(self.deblurred_frames_generator))
            cv2.imwrite(os.path.join(directory, f"s_{i}.png"), frame1)
            i += 1
            while True:
                frame2 = self.transform(next(self.deblurred_frames_generator))
                if MS_SSIM_L1_LOSS(frame1, frame2) > self.threshold:
                    cv2.imwrite(os.path.join(directory, f"s_{i}.png"), frame2)
                    i += 1
                frame1 = frame2
        except StopIteration:
            print(f"Saved {i} number of key frames")

    def transform(self, frame):
        frame = self.bgr_to_rgb(frame)
        frame = ToTensor()(frame)
        return frame

    def display(self):
        """
        Display the deblurred frames of the video
        """
        self.get_generators()
        try:
            for frame in video.deblurred_frames:
                plt.imshow(self.bgr_to_rgb(frame))
                plt.show()
        except KeyboardInterrupt:
            print("Exiting...")
            sys.exit()
        self.reset()


    def mkdir(self, directory):
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


    def __len__(self):
        """
        return the number of deblurred frames of the video
        """
        self.get_generators()
        len = sum(1 for _ in self.deblurred_frames_generator)
        self.reset()
        return len

    def reset(self):
        self.deblurred_frames_generator = None

if __name__ == "__main__":
    start_time = time.time()
    VID = 'V1'
    video = Video(path="../VIDEOS/" + VID +".mp4", per_second_acquisition=5)
    print(f'number of deblurred frames: {len(video)}')
    video.save(directory="../VIDEOS/deblurred/" + VID + "/", type='blur')
    del video
    print(f"--- {time.time() - start_time} seconds ---")
    # displaying the deblurred frames
    # video.display()
