import cv2
import matplotlib.pyplot as plt
import numpy as np


class StableframeExtractor:
    """ Extract a stable frame from the video stream"""

    def __init__(self, video_path, viz_image):
        self.video_path = video_path
        self.viz_image = viz_image

    def StableframeExtractor(self):
        videocap = cv2.VideoCapture(self.video_path)
        if not videocap.isOpened():
            print("Error opening video file")
        prev_frame = None
        buf_stable_frame = None
        stable_frame = None

        length = int(videocap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_n in range(length):
            success, cur_frame = videocap.read()
            if success:
                buf_stable_frame = cur_frame
                cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
                cur_frame = cv2.resize(cur_frame, (1024, 1024))
                cur_frame = cv2.blur(cur_frame, (5, 5))

                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, cur_frame)
                    unique = len(np.unique(diff))
                    print("Frame No:", frame_n, " | ", unique)

                    if 400 < unique < 15:
                        print(" The stable frame number is : " + str(frame_n))
                        stable_frame_count = frame_n
                        stable_frame = buf_stable_frame
                        break
                prev_frame = cur_frame

        if self.viz_image:
            plt.imshow(stable_frame)

        return stable_frame
