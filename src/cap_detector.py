import argparse
import csv
import glob
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import tensorflow as tf

from cv_object_detector import cv_Detector
from tf2_object_detector import tf2_Detector


class cap_detector:
    def __init__(self, video_path: str, result_path: str, frozengraph_path: str,
                 config_path: str, ckpt_path: str, tf_version: int):
        self.video_path = video_path
        self.video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.result_path = result_path
        self.frozen_graph = frozengraph_path
        self.pbtxt = config_path
        self.threshold = 0.41
        self.tf_version = tf_version
        self.ckpt_path = ckpt_path

        self.label_dict = {1: "Tray",
                           2: "BottleCap_FaceUp",
                           3: "BottleCap_FaceDown",
                           4: "BottleCap_Deformed"}
        if tf_version == 2:
            self.cap_det = tf2_Detector(self.label_dict, self.ckpt_path,
                                        self.threshold)
        else:
            self.cap_det = cv_Detector(self.label_dict, self.frozen_graph,
                                       self.pbtxt)

    def StableframeExtractor(self, viz_image=False):

        videocap = cv2.VideoCapture(self.video_path)
        if not videocap.isOpened():
            print("Error opening video file")
        prev_frame = None
        buf_stable_frame = None
        prev_unique = 0
        rising_edge = False
        falling_edge = False

        length = int(videocap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_n in range(length):
            success, cur_frame = videocap.read()
            if success:
                buf_stable_frame = cur_frame
                cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
                cur_frame = cv2.resize(cur_frame, (1024, 1024))
                cur_frame = cv2.blur(cur_frame, (5, 5))
                if frame_n == int(length / 2):
                    stable_frame = buf_stable_frame
                    self.frame_number = frame_n

                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, cur_frame)
                    unique = len(np.unique(diff))
                    if unique - prev_unique > 30:
                        rising_edge = True
                    if prev_unique - unique > 30:
                        falling_edge = True
                    print("Frame No:", frame_n, " | ", unique)

                    if unique < 30 and rising_edge and falling_edge:
                        self.frame_number = frame_n
                        print(" The stable frame number is : " + str(frame_n))
                        stable_frame_count = frame_n
                        stable_frame = buf_stable_frame
                        break
                    prev_unique = unique
                prev_frame = cur_frame

        if viz_image:
            img_out = os.path.join(self.result_path, str(self.video_name +
                                                         '.png'))
            print(img_out)
            cv2.imwrite(img_out, stable_frame)
        return stable_frame

    def bbox_in_check(self, in_box):
        tray = self.cap_det.tray
        if in_box[0] > tray[0] and in_box[1] > tray[1]:
            if in_box[2] < tray[2] and in_box[3] < tray[3]:
                return True
        else:
            print("Box not in the Tray")
            return False

    def filter_detections(self, boxes_list, overlap_threshold=0.9):
        filtered_boxes = []
        for idx in range(len(boxes_list)):
            x_min = int(boxes_list[idx][0])
            y_min = int(boxes_list[idx][1])
            x_max = int(boxes_list[idx][2])
            y_max = int(boxes_list[idx][3])
            cls = str(boxes_list[idx][4])
            if cls == "Tray":
                continue
            score = str(np.round(boxes_list[idx][-1], 2))
            box = [x_min, y_min, x_max, y_max, cls, float(score)]
            if self.bbox_in_check(box):
                filtered_boxes.append(box)
        return filtered_boxes

    def perform_inference(self, viz_save=False, to_csv=True):
        self.stable_frame = self.StableframeExtractor(viz_image=False)

        if self.tf_version == 2:
            self.det_array = self.cap_det.detect_from_image(self.stable_frame)
        else:
            image = self.cap_det.img_resizer(self.stable_frame, op_size=(1024,
                                                                         1024))
            detections = self.cap_det.detect_from_image(image=self.stable_frame)
            self.det_array = self.cap_det.provide_output(detections)

        filtered_boxes = self.filter_detections(self.det_array)
        img = self.display_detections(self.stable_frame, filtered_boxes,
                                      det_time=None)
        if to_csv:
            with open(os.path.join(self.result_path.rstrip(),
                                   str(self.video_name + ".csv")), 'w',
                      newline='') as f:
                writer = csv.writer(f, delimiter=',')
                all_results_array = []
                for arr in filtered_boxes:
                    result_str = ''
                    if arr[4] != "Tray":
                        result_str = result_str + str(self.frame_number) + ','
                        x_mid, y_mid = arr[0] + arr[2] / 2, arr[1] + arr[3] / 2
                        result_str = result_str + str(int(x_mid)) + ','
                        result_str = result_str + str(int(y_mid)) + ','
                        result_str = result_str + (arr[4])
                        all_results_array.append(result_str)
                writer.writerows([[item] for item in all_results_array])

        if viz_save:
            cv2.imshow('TF2 Detection', img)
            img_out = os.path.join(self.result_path.rstrip(),
                                   str(self.video_name + ".png"))
            cv2.imwrite(img_out, img)
            cv2.waitKey(0)

    def display_detections(self, image, boxes_list, det_time=None):
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
            cv2.rectangle(img, (x_min, y_min - 20), (x_min, y_min), (255, 255,
                                                                     255), -1)
            cv2.putText(img, text, (x_min + 5, y_min - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if det_time is not None:
            fps = round(1000. / det_time, 1)
            fps_txt = str(fps) + " FPS"
            cv2.putText(img, fps_txt, (25, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--video_path', action='store', type=str,
                        help='Path to Videos collected')
    parser.add_argument('--result_path', action='store', type=str,
                        help='Path in which results are stored')
    parser.add_argument('--to_show', action='store', default=False, type=bool,
                        help='Path to the config .pbtxt file')

    args = parser.parse_args()
    dirname = os.path.dirname(__file__)

    video_path = args.video_path
    result_path = args.result_path
    frozengraph_path = '../models/tf1_model/frozen_inference_graph.pb'
    config_path = '../models/tf1_model/sample.pbtxt'
    ckpt_path = os.path.join(dirname, '../models/tf2_model/saved_model/')
    to_show = args.to_show

    version = tf.__version__
    tf_version = int(version[0])
    # tf_version = 2

    videos = glob.glob(os.path.join(video_path, '*.mp4'))
    print("Number of videos present are {}".format(len(videos)))
    for video in videos:
        cap_det_model = cap_detector(video_path=video, result_path=result_path,
                                     frozengraph_path=frozengraph_path,
                                     config_path=config_path,
                                     ckpt_path=ckpt_path, tf_version=tf_version)
        cap_det_model.perform_inference(viz_save=False, to_csv=True)
