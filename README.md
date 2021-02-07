# BCDC-Net
A Project for Computer Vision Course

## Steps of execution

1) Edit **ubuntu_run.sh** inside the BCDC-Net folder

2) Check if the model is present in **models/tf2_model/**

    The **tf2_model/** folder contains 1. checkpoint (folder)  2. pipeline.config (config file) 3. saved_model (folder)

3) The command should be as follows 

    python src/cap_detector.py --video_path PATH_TO_VIDEOS --result_path PATH_TO_RESULTS_FOLDER

    Replace **PATH_TO_VIDEOS** with the path in which the test videos are present in double quotes("").

    Replace **PATH_TO_RESULTS_FOLDER** with the path in which the result csv files are to be stored in double quotes("").
