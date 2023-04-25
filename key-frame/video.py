import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import time


class Video:
    def __init__(self, path, per_second_acquisition=5):
        """
        class to extract frames from a video

        Parameters:
        path (str): The path to the video.
        per_second_acquisition (int): The number of frames to be extracted per second.
        remove_blur (bool): Whether to remove blur from the frames or not.

        Attributes:
        frames (tuple): A tuple containing the frames of the video.
        deblurred_frames (tuple): A tuple containing the deblurred frames of the video.
        """
        # frames_generator is a generator object that yields the frames of the video
        self.frames_generator = self.extract(cv2.VideoCapture(path), per_second_acquisition)
        deblurred_frames_generator = self.deblur(per_second_acquisition)
        self.deblurred_frames = tuple(deblurred_frames_generator)

    def frame_rate(self, video):
        """
        Get the frame rate of a video.

        Parameters:
        video (cv2.VideoCapture): A cv2.VideoCapture object representing the video.

        Returns:
        float: The frame rate of the video.
        """
        return video.get(cv2.CAP_PROP_FPS)

    def extract(self, video, per_second_acquisition):
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
        collect_every = fps // per_second_acquisition #Raiseerror if fps < per_second_acquisition
        while True:
            success, frame = video.read()
            if success:
                if current_frame % collect_every == 0:
                    yield self.bgr_to_rgb(frame)
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

    def deblur(self, per_second_acquisition):
        max_laplacian_var = 0
        count = 0
        max_laplacian_var_frame = None
        for frame in self.frames_generator:
            if max_laplacian_var < cv2.Laplacian(frame, cv2.CV_64F).var():
                max_laplacian_var = cv2.Laplacian(frame, cv2.CV_64F).var()
                max_laplacian_var_frame = frame
            count += 1
            if count % per_second_acquisition == 0:
                yield max_laplacian_var_frame
                max_laplacian_var = 0
                count = 0
"""
    def __len__(self):
        self.remove_blur = False
        if self.remove_blur:
            return sum(1 for _ in self.deblurred_frames)
        else:
            return sum(1 for _ in self.frames_generator)
"""

if __name__ == "__main__":
    start_time = time.time()
    video = Video(path="../VIDEOS/V0.mp4", per_second_acquisition=5)
    print(f'length of deblurred frames: {len(video.deblurred_frames)}')
    print(f"Time taken to extract deblurred frames: {time.time() - start_time} seconds")
    """
    Printing the frames of the deblurred video
    Least blurry frame every second is selected
    """
    for frame in video.deblurred_frames:
        plt.imshow(frame)
        plt.show()
