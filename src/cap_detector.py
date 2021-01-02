import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# from src.object_detector import Image_Detector
from object_detector import Image_Detector


class cap_detector:

    def __init__(self, video_path, result_path, frozengraph_path, config_path):
        self.video_path = video_path
        self.video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.result_path = result_path
        self.frozen_graph = frozengraph_path
        self.pbtxt = config_path

        self.label_dict = {0: "Tray",
                           1: "BottleCapFaceUp",
                           2: "BottleCapFaceDown",
                           3: "BottleCapDeformed"}

        self.cap_det = Image_Detector(self.label_dict, self.frozen_graph, self.pbtxt)

    def StableframeExtractor(self, viz_image=True):

        videocap = cv2.VideoCapture(self.video_path)
        if not videocap.isOpened():
            print("Error opening video file")
        prev_frame = None
        buf_stable_frame = None
        # self.stable_frame = None

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
                    self.stable_frame = buf_stable_frame

                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, cur_frame)
                    unique = len(np.unique(diff))
                    print("Frame No:", frame_n, " | ", unique)

                    if unique < 30 and frame_n > 85:
                        print(" The stable frame number is : " + str(frame_n))
                        stable_frame_count = frame_n
                        self.stable_frame = buf_stable_frame
                        break
                prev_frame = cur_frame

        if viz_image:
            img_out = os.path.join(self.result_path, self.video_name + '.png')
            cv2.imwrite(img_out, self.stable_frame)
        return self.stable_frame

    def perform_inference(self, viz_save=True, to_csv=False):
        self.stable_frame = self.StableframeExtractor(viz_image=True)
        image = self.cap_det.img_resizer(self.stable_frame, op_size=(600, 1024))
        detections = self.cap_det.detect_from_image(image=image)
        self.det_array = self.cap_det.provide_output(detections)

        print(self.det_array)
        if viz_save:
            img = self.display_detections(self.stable_frame, self.det_array, det_time=True)
            cv2.imshow('TF2 Detection', img)
            cv2.waitKey(0)

            img_out = os.path.join(self.result_path, self.video_name + '.png')
            cv2.imwrite(img_out, img)

    def display_detections(self, image, boxes_list, det_time=True):
        if not boxes_list:
            print("No objects Found")
            return image  # input list is empty
        img = image.copy()
        for idx in range(len(boxes_list)):
            x_min = boxes_list[idx][0]
            y_min = boxes_list[idx][1]
            x_max = boxes_list[idx][2]
            y_max = boxes_list[idx][3]
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
    video_path = '/media/jarvis/CommonFiles/4th_semester/CV/CV_Project/test_dataset/videos/'
    result_path = '/media/jarvis/CommonFiles/4th_semester/CV/CV_Project/test_dataset/output_images/'
    frozengraph_path = '../models/frozen_inference_graph.pb'
    config_path = '../models/output.pbtxt'
    videos = glob.glob(os.path.join(video_path, '*.mp4'))
    for video in videos:
        cap_det_model = cap_detector(video_path=video, result_path=result_path, frozengraph_path=frozengraph_path,
                                     config_path=config_path)
        cap_det_model.perform_inference()

