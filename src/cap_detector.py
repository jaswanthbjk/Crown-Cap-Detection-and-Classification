import argparse
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import tensorflow as tf

# from src.object_detector import Image_Detector
from cv_object_detector import cv_Detector
from tf2_object_detector import tf2_Detector


# from src.cv_object_detector import cv_Detector
# from src.tf2_object_detector import tf2_Detector


class cap_detector:
    def __init__(self, video_path: str, result_path: str, frozengraph_path: str, config_path: str, ckpt_path: str,
                 tf_version: int):
        self.video_path = video_path
        self.video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.result_path = result_path
        self.frozen_graph = frozengraph_path
        self.pbtxt = config_path
        self.threshold = 0.41
        self.tf_version = tf_version
        self.ckpt_path = ckpt_path

        self.label_dict = {1: "Tray",
                           2: "BottleCapFaceUp",
                           3: "BottleCapFaceDown",
                           4: "BottleCapDeformed"}
        if tf_version == 2:
            self.cap_det = tf2_Detector(self.label_dict, self.ckpt_path, self.threshold)
        else:
            self.cap_det = cv_Detector(self.label_dict, self.frozen_graph, self.pbtxt)
        # self.cap_det = cv_Detector(self.label_dict, self.frozen_graph, self.pbtxt)

    def StableframeExtractor(self, viz_image=True):

        videocap = cv2.VideoCapture(self.video_path)
        if not videocap.isOpened():
            print("Error opening video file")
        prev_frame = None
        buf_stable_frame = None

        length = int(videocap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(length)

        for frame_n in range(length):
            success, cur_frame = videocap.read()
            if success:
                buf_stable_frame = cur_frame
                cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
                cur_frame = cv2.resize(cur_frame, (1024, 1024))
                cur_frame = cv2.blur(cur_frame, (5, 5))
                if frame_n == int(length / 2):
                    stable_frame = buf_stable_frame

                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, cur_frame)
                    unique = len(np.unique(diff))
                    print("Frame No:", frame_n, " | ", unique)

                    if unique < 30 and frame_n > 85:
                        print(" The stable frame number is : " + str(frame_n))
                        stable_frame_count = frame_n
                        stable_frame = buf_stable_frame
                        break
                prev_frame = cur_frame

        if viz_image:
            img_out = os.path.join(self.result_path, self.video_name + '.png')
            cv2.imwrite(img_out, stable_frame)
        return stable_frame

    def perform_inference(self, viz_save=False, to_csv=False):
        self.stable_frame = self.StableframeExtractor(viz_image=True)

        if self.tf_version == 2:
            self.det_array = self.cap_det.DetectFromImage(self.stable_frame)
        else:
            image = self.cap_det.img_resizer(self.stable_frame, op_size=(640, 580))
            detections = self.cap_det.detect_from_image(image=self.stable_frame)
            self.det_array = self.cap_det.provide_output(detections)

        print(self.det_array)
        img = self.display_detections(self.stable_frame, self.det_array, det_time=True)
        if viz_save:
            cv2.imshow('TF2 Detection', img)
            cv2.waitKey(0)

        img_out = os.path.join(self.result_path, self.video_name + '.png')
        cv2.imwrite(img_out, img)

    def display_detections(self, image, boxes_list, det_time=True):
        image = self.stable_frame
        image_w, image_h = image.shape[0], image.shape[1]
        # scale_w, scale_h = image_w / 300.0, image_h / 300.0
        scale_w, scale_h = 1, 1
        if not boxes_list:
            print("No objects Found")
            return image  # input list is empty
        img = image.copy()
        for idx in range(len(boxes_list)):
            x_min = int(boxes_list[idx][0] * scale_h)
            y_min = int(boxes_list[idx][1] * scale_w)
            x_max = int(boxes_list[idx][2] * scale_h)
            y_max = int(boxes_list[idx][3] * scale_w)
            cls = str(boxes_list[idx][4])
            score = str(np.round(boxes_list[idx][-1], 2))

            text = cls + ": " + score
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            cv2.rectangle(img, (x_min, y_min - 20), (x_min, y_min), (255, 255, 255), -1)
            cv2.putText(img, text, (x_min + 5, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if det_time is not None:
            fps = round(1000. / det_time, 1)
            fps_txt = str(fps) + " FPS"
            cv2.putText(img, fps_txt, (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--video_path', action='store', type=str, help='Path to Videos collected')
    parser.add_argument('--result_path', action='store', type=str, help='Path in which results are stored')
    parser.add_argument('--frozen_graph_path', action='store', type=str, help='Path to the Frozen graph .pb file')
    parser.add_argument('--config_path', action='store', type=str, help='Path to the config .pbtxt file')
    parser.add_argument('--ckpt_path', action='store', type=str, help='Path to the config .pbtxt file')
    parser.add_argument('--to_show', action='store', default=False, type=bool, help='Path to the config .pbtxt file')

    args = parser.parse_args()
    print(args)

    video_path = args.video_path
    result_path = args.result_path
    frozengraph_path = args.frozen_graph_path
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    to_show = args.to_show

    version = tf.__version__
    # tf_version = int(version[0])
    tf_version = 2

    videos = glob.glob(os.path.join(video_path, '*.mp4'))
    print("Number of videos present are {}".format(len(videos)))
    for video in videos:
        cap_det_model = cap_detector(video_path=video, result_path=result_path, frozengraph_path=frozengraph_path,
                                     config_path=config_path, ckpt_path=ckpt_path, tf_version=tf_version)
        cap_det_model.perform_inference(viz_save=to_show)
