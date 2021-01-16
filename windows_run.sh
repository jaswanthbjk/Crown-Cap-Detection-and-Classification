#!/bin/bash

cd src/

python cap_detector.py --video_path F:\\4th_semester\\CV\\CV_Project\\test_dataset\\videos\\ --result_path F:\\4th_semester\\CV\\CV_Project\\test_dataset\\output_images\\ --frozen_graph_path ../models/tf1_model/frozen_inference_graph.pb --config_path ../models/tf1_model/sample.pbtxt --ckpt_path ../models/tf2_model/saved_model