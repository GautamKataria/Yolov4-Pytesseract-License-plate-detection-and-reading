# Deep-learning-model-to-detect license plates/ Number plates of a car and read them-in-realtime using a custom trained yolov4 model and Tesseract-OCR (please read before executing)

### Before starting with this,
Dont forget to download the preequisites to using the yolov4 model on your local machine in my YOLOV4 Tutorials 
repository link --> [Prerequisites to yolov4](https://github.com/GautamKataria/YOLOv4-Tutorials)

#### MOVE the obj-license.data and custom cfg file into the cfg folder in the x64 folder in your darknet build.

#### MOVE the obj.names into the data folder in the x64 folder in your darknet build.

#### Also download Tesseract from here [Link to download tesseract](https://github.com/UB-Mannheim/tesseract/wiki)

#### Dont forget to download tesseract and change the path in the python file to the location of tesseract.exe in your local machine.

Download the weights from my google drive.  LINK --> [Weights here](https://drive.google.com/file/d/1Ld_sv4tDPUISv1rxYE3XjSKJQPvHCiyt/view?usp=sharing)

The Yolo-v4 model is custom trained on license plates from the Google open images dataset. LINK--> [here](https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=detection&c=%2Fm%2F01jfm_)

## Working:-

##### Our yolov4 model is custom trained on license plates from the google open image dataset [Dataset for yolov4 training](https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=detection&c=%2Fm%2F01jfm_)
##### We use opencv to feed the video into the yolov4 model which feeds it to the custom trained yolov4 model which inturn gives us bounding boxes for the license plates.
##### Each bounding box is then cropped from the frame and passed on. 
##### After some processing using opencv the images are passed onto the tesseract OCR which inturn outputs the license plate / number plate  number
##### The output of the yolov4 detections will be saved in the demo folder in the x64 folder in your darknet build.
