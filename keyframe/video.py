import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
from MS_SSIM_L1_loss import MS_SSIM_L1_LOSS
from processing import Helper


class Video:
    def __init__(self, path, per_second_acquisition=5, threshold=60, use_model=True):
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
        self.use_model = use_model

    def get_generators(self):
        """
        Create object generators from the video.

        Attributes:
        frames_generator (generator): A generator object that yields the frames of the video.
        """
        frames_generator = self.extract(cv2.VideoCapture(self.path))
        deblurred_frames_generator = self.deblur(frames_generator)
        key_frames_generator = self.key_filter(deblurred_frames_generator)
        frames_generator = None
        return deblurred_frames_generator, key_frames_generator

    def save(self,directory, type = 'blur'):
        """
        Save the frames of the video in a directory.

        Parameters:
        directory (str): The path to the directory where the frames are to be saved.
        type (str): The type of frames to be saved. 'blur' for deblurred frames and 'key' for key frames.

        Returns:
        None
        """
        Helper.mkdir(directory)
        deblurred_frames_generator,key_frames_generator =self.get_generators()
        if type == 'blur':
            self.save_frames(directory, deblurred_frames_generator)
        elif type == 'key':
            self.save_frames(directory, key_frames_generator)

    def save_frames(self, directory, frames_generator):
        """
        Save the frames of the video in a directory.

        Parameters:
        directory (str): The path to the directory where the frames are to be saved.
        frames_generator (generator): A generator object that yields the frames of the video.

        Returns:
        None
        """
        for i, frame in enumerate(frames_generator):
            cv2.imwrite(os.path.join(directory, f"s_{i}.png"), frame)
        print(f"Saved {i+1} number of frames")

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
        fps = Helper.frame_rate(video)
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

    def deblur(self, frames_generator):
        """
        Generator that yields deblurred frames of a video.

        Parameters:
        frames_generator (generator): A generator object that yields the frames of the video.

        Returns:
        generator: A generator object that yields the deblurred frames of the video.
        """
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
        print("Deblurred frames generated")

    def key_filter(self, deblurred_frames_generator):
        """
        Generator that yields key frames of a video.

        Returns:
        generator: A generator object that yields the key frames of the video.
        """
        model = Helper.load_model(self.use_model)
        diff = MS_SSIM_L1_LOSS()
        try:
            frame1 = next(deblurred_frames_generator)
            yield frame1
            while True:
                frame2 = next(deblurred_frames_generator)
                if diff(model(Helper.transform(frame1)), model(Helper.transform(frame2))).item() > self.threshold:
                    yield frame2
                frame1 = frame2
        except StopIteration:
            print(f"Key frames generated")

    def display(self, type = 'blur'):
        """
        Display the deblurred frames of the video
        """
        deblurred_frames_generator,key_frames_generator =self.get_generators()
        if type == 'blur':
            frames_generator = deblurred_frames_generator
        elif type == 'key':
            frames_generator = key_frames_generator
        try:
            for frame in frames_generator:
                plt.imshow(Helper.bgr_to_rgb(frame))
                plt.show()
        except KeyboardInterrupt:
            print("Exiting...")
            sys.exit()

    def __len__(self, type = 'blur'):
        """
        return the number of deblurred frames of the video
        """
        deblurred_frames_generator, key_frames_generator =self.get_generators()
        if type == 'blur':
            frames_generator = deblurred_frames_generator
        elif type == 'key':
            frames_generator = key_frames_generator
        len = sum(1 for _ in frames_generator)
        return len


if __name__ == "__main__":
    start_time = time.time()
    VID = 'V1'
    video = Video(path="../VIDEOS/" + VID +".mp4", per_second_acquisition=5)
    print(f'number of deblurred frames: {len(video)}')
    print(f"--- {time.time() - start_time} seconds ---")
    video.save(directory="../VIDEOS/deblurred/" + VID + "/", type='blur')
    del video
    print(f"--- {time.time() - start_time} seconds ---")
    # displaying the deblurred frames
    # video.display()
