import cv2
import numpy as np
import tensorflow as tf


class tf2_Detector:
    def __init__(self, label_dict: dict, path_to_checkpoint: str, threshold: float = 0.5):

        self.Threshold = threshold
        self.label_dict = label_dict
        tf.keras.backend.clear_session()
        self.detect_fn = tf.saved_model.load(path_to_checkpoint)

    def tray_detector(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 205, 1)
        contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        filtered = []
        for c in contours:
            if cv2.contourArea(c) < 1000:
                continue
            filtered.append(c)

        filtered = sorted(filtered, key=lambda c: cv2.contourArea(c), reverse=True)
        rect = cv2.boundingRect(filtered[0])
        x, y, w, h = rect
        return [x, y, x + w, y + h, 'Tray', 100]

    def DetectFromImage(self, img):
        self.image = img
        im_height, im_width, _ = img.shape
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        input_tensor = np.expand_dims(img, 0)
        detections = self.detect_fn(input_tensor)

        bboxes = detections['detection_boxes'][0].numpy()
        bclasses = detections['detection_classes'][0].numpy().astype(np.int32)
        bscores = detections['detection_scores'][0].numpy()
        det_boxes = self.ExtractBBoxes(bboxes, bclasses, bscores, im_width, im_height)

        return det_boxes

    def ExtractBBoxes(self, bboxes, bclasses, bscores, im_width, im_height):
        bbox = []
        seen_tray = False
        for idx in range(len(bboxes)):
            if bscores[idx] >= self.Threshold:
                y_min = int(bboxes[idx][0] * im_height)
                x_min = int(bboxes[idx][1] * im_width)
                y_max = int(bboxes[idx][2] * im_height)
                x_max = int(bboxes[idx][3] * im_width)
                class_label = self.label_dict[int(bclasses[idx])]
                if class_label == 'Tray':
                    seen_tray = True
                bbox.append([x_min, y_min, x_max, y_max, class_label, float(bscores[idx])])
        if not seen_tray:
            bbox.append(self.tray_detector(self.image))
        return bbox

    def DisplayDetections(self, image, boxes_list, det_time=None):
        if not boxes_list: return image  # input list is empty
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

        if det_time != None:
            fps = round(1000. / det_time, 1)
            fps_txt = str(fps) + " FPS"
            cv2.putText(img, fps_txt, (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        return img
