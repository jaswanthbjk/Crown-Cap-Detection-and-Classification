import cv2
import numpy as np


class cv_Detector:
    """ Detector class for performing
    1) resizing to required image size
    2) Perform inference on a new image using the trained network
    Args:
        label_dict: Dictionary of class_id mapped to class_names
        frozen_graph: Tensorflow frozen graph of trained detection model
        pbtxt: configuration file of the model choosen

    Outputs:
        result: list of lists, every list representing a bounding box for the caps present in the image
        final_image: Image with bounding boxes drawn on it """

    def __init__(self, label_dict: dict, frozen_graph: str, pbtxt: str):
        self.save_output = True
        self.Threshold = 0.3
        self.label_dict = label_dict
        self.Net = cv2.dnn.readNetFromTensorflow(model=frozen_graph, config=pbtxt)

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

    def img_resizer(self, image, op_size):
        """Resize the input Image to required size
        Args:
            op_size: size to which the input has to be resized
        output:
            resized Image: Image after resizing """
        self.resize = op_size
        self.resized_image = cv2.resize(image, self.resize, interpolation=cv2.INTER_AREA)
        return self.resized_image

    def detect_from_image(self, image):
        """ Perform Inferencing
        Args:
            image: Input to the detection model
        outputs:
            detection: All the bounding boxes inferenced by the model"""
        self.image = image
        self.image_h, self.image_w = np.shape(self.image)[0], np.shape(self.image)[1]
        self.Net.setInput(cv2.dnn.blobFromImage(self.resized_image, size=self.resize,
                                                swapRB=True, crop=True))
        print('Graph loaded')
        detections = self.Net.forward()
        return detections

    def provide_output(self, detections):
        """Re-arrange the model outputs into understandable values"""
        self.result_array = list()
        seen_tray = False
        for detection in detections[0, 0, :, :]:
            score = float(detection[2])
            if score > self.Threshold:
                x_min = int(detection[3] * self.image_w)
                y_min = int(detection[4] * self.image_h)
                x_max = int(detection[5] * self.image_w)
                y_max = int(detection[6] * self.image_h)
                cls_label = self.label_dict[int(detection[1])]
                if cls_label == 'Tray':
                    seen_tray = True
                single_result = [x_min, y_min, x_max, y_max, cls_label, float(score)]
                self.result_array.append(single_result)
        if not seen_tray:
            self.result_array.append(self.tray_detector(self.image))

        return self.result_array

    def show_save_image(self, save_output: bool, output_dir: str):
        """ To display the image after bounding box marking
        Args:
            save_output: Flag for saving the generated image or not
            output_dir: Path in which the output should be saved
            """
        final_img = self.result_array.copy()
        if not self.result_array:
            return final_img
        else:
            for image_id in range(len(self.result_array)):
                x_min = self.result_array[image_id][0]
                y_min = self.result_array[image_id][1]
                x_max = self.result_array[image_id][2]
                y_max = self.result_array[image_id][3]
                cls = str(self.result_array[image_id][4])
                score = str(np.round(self.result_array[image_id][-1], 2))

                text = cls + ": " + score
                cv2.rectangle(final_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
                cv2.rectangle(final_img, (x_min, y_min - 20), (x_min, y_min), (255, 255, 255), -1)
                cv2.putText(final_img, text, (x_min + 5, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 1)

        if save_output:
            cv2.imwrite(output_dir, final_img)

        return final_img