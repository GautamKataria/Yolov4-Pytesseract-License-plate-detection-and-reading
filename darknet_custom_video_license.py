from ctypes import *                                               # Import libraries
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
from keras.models import load_model
import keyboard
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

threshold1 = 0.75
#model = load_model('model_30.h5')
w1 = 640
h1 = 480

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img


def getCalssName(classNo):
    if classNo == 0:
        return 'Speed Limit 20 km/h'
    elif classNo == 1:
        return 'Speed Limit 30 km/h'
    elif classNo == 2:
        return 'Speed Limit 50 km/h'
    elif classNo == 3:
        return 'Speed Limit 60 km/h'
    elif classNo == 4:
        return 'Speed Limit 70 km/h'
    elif classNo == 5:
        return 'Speed Limit 80 km/h'
    elif classNo == 6:
        return 'End of Speed Limit 80 km/h'
    elif classNo == 7:
        return 'Speed Limit 100 km/h'
    elif classNo == 8:
        return 'Speed Limit 120 km/h'
    elif classNo == 9:
        return 'No passing'
    elif classNo == 10:
        return 'No passing for vechiles over 3.5 metric tons'
    elif classNo == 11:
        return 'Right-of-way at the next intersection'
    elif classNo == 12:
        return 'Priority road'
    elif classNo == 13:
        return 'Yield'
    elif classNo == 14:
        return 'Stop'
    elif classNo == 15:
        return 'No vechiles'
    elif classNo == 16:
        return 'Vechiles over 3.5 metric tons prohibited'
    elif classNo == 17:
        return 'No entry'
    elif classNo == 18:
        return 'General caution'
    elif classNo == 19:
        return 'Dangerous curve to the left'
    elif classNo == 20:
        return 'Dangerous curve to the right'
    elif classNo == 21:
        return 'Double curve'
    elif classNo == 22:
        return 'Bumpy road'
    elif classNo == 23:
        return 'Slippery road'
    elif classNo == 24:
        return 'Road narrows on the right'
    elif classNo == 25:
        return 'Road work'
    elif classNo == 26:
        return 'Traffic signals'
    elif classNo == 27:
        return 'Pedestrians'
    elif classNo == 28:
        return 'Children crossing'
    elif classNo == 29:
        return 'Bicycles crossing'
    elif classNo == 30:
        return 'Beware of ice/snow'
    elif classNo == 31:
        return 'Wild animals crossing'
    elif classNo == 32:
        return 'End of all speed and passing limits'
    elif classNo == 33:
        return 'Turn right ahead'
    elif classNo == 34:
        return 'Turn left ahead'
    elif classNo == 35:
        return 'Ahead only'
    elif classNo == 36:
        return 'Go straight or right'
    elif classNo == 37:
        return 'Go straight or left'
    elif classNo == 38:
        return 'Keep right'
    elif classNo == 39:
        return 'Keep left'
    elif classNo == 40:
        return 'Roundabout mandatory'
    elif classNo == 41:
        return 'End of no passing'
    elif classNo == 42:
        return 'End of no passing by vechiles over 3.5 metric tons'


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    # Colored labels dictionary
    color_dict = {
        'person' : [0, 255, 255],'Traffic sign': [0,0,255], 'bicycle': [238, 123, 158], 'car' : [24, 245, 217], 'motorbike' : [224, 119, 227],
        'aeroplane' : [154, 52, 104], 'bus' : [179, 50, 247], 'train' : [180, 164, 5], 'truck' : [82, 42, 106],
        'boat' : [201, 25, 52], 'traffic light' : [62, 17, 209], 'fire hydrant' : [60, 68, 169], 'stop sign' : [199, 113, 167],
        'parking meter' : [19, 71, 68], 'bench' : [161, 83, 182], 'bird' : [75, 6, 145], 'cat' : [100, 64, 151],
        'dog' : [156, 116, 171], 'horse' : [88, 9, 123], 'sheep' : [181, 86, 222], 'cow' : [116, 238, 87],'elephant' : [74, 90, 143],
        'bear' : [249, 157, 47], 'zebra' : [26, 101, 131], 'giraffe' : [195, 130, 181], 'backpack' : [242, 52, 233],
        'umbrella' : [131, 11, 189], 'handbag' : [221, 229, 176], 'tie' : [193, 56, 44], 'suitcase' : [139, 53, 137],
        'frisbee' : [102, 208, 40], 'skis' : [61, 50, 7], 'snowboard' : [65, 82, 186], 'sports ball' : [65, 82, 186],
        'kite' : [153, 254, 81],'baseball bat' : [233, 80, 195],'baseball glove' : [165, 179, 213],'skateboard' : [57, 65, 211],
        'surfboard' : [98, 255, 164],'tennis racket' : [205, 219, 146],'bottle' : [140, 138, 172],'wine glass' : [23, 53, 119],
        'cup' : [102, 215, 88],'fork' : [198, 204, 245],'knife' : [183, 132, 233],'spoon' : [14, 87, 125],
        'bowl' : [221, 43, 104],'banana' : [181, 215, 6],'apple' : [16, 139, 183],'sandwich' : [150, 136, 166],'orange' : [219, 144, 1],
        'broccoli' : [123, 226, 195],'carrot' : [230, 45, 209],'hot dog' : [252, 215, 56],'pizza' : [234, 170, 131],
        'donut' : [36, 208, 234],'cake' : [19, 24, 2],'chair' : [115, 184, 234],'sofa' : [125, 238, 12],
        'pottedplant' : [57, 226, 76],'bed' : [77, 31, 134],'diningtable' : [208, 202, 204],'toilet' : [208, 202, 204],
        'tvmonitor' : [208, 202, 204],'laptop' : [159, 149, 163],'mouse' : [148, 148, 87],'remote' : [171, 107, 183],
        'keyboard' : [33, 154, 135],'cell phone' : [206, 209, 108],'microwave' : [206, 209, 108],'oven' : [97, 246, 15],
        'toaster' : [147, 140, 184],'sink' : [157, 58, 24],'refrigerator' : [117, 145, 137],'book' : [155, 129, 244],
        'clock' : [53, 61, 6],'vase' : [145, 75, 152],'scissors' : [8, 140, 38],'teddy bear' : [37, 61, 220],
        'hair drier' : [129, 12, 229],'toothbrush' : [11, 126, 158]
    }
    
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        name_tag = str(detection[0].decode())
        for name_key, color_val in color_dict.items():
            if name_key == name_tag:
                color = color_val 
                xmin, ymin, xmax, ymax = convertBack(
                float(x), float(y), float(w), float(h))
                pt1 = (xmin, ymin)
                pt2 = (xmax, ymax)
                cv2.rectangle(img, pt1, pt2, color, 1)
                cv2.putText(img,
                            detection[0].decode() +
                            " [" + str(round(detection[1] * 100, 2)) + "]",
                            (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color, 2)
    return img


netMain = None
metaMain = None
altNames = None


def YOLO(in123, in1234):
   
    global metaMain, netMain, altNames
    configPath = "./cfg/yolov4-obj-license.cfg"                                 # Path to cfg
    weightPath = "./custom_license.weights"                                 # Path to weights
    metaPath = "./cfg/obj-license.data"                                    # Path to meta data
    #in123 = input("Enter name of mp4 video saved in data folder without its extension: ")
    #in1234 = F'{in123}_output'

    if not os.path.exists(configPath):                              # Checks whether file exists otherwise return ValueError
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:                                             # Checks the metaMain, NetMain and altNames. Loads it in script
        netMain = darknet.load_net_custom(configPath.encode( 
            "ascii"), weightPath.encode("ascii"), 0, 1)             # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)                                      # Uncomment to use Webcam

    cap = cv2.VideoCapture(F"./data/{in123}.mp4")                             # Local Stored video detection - Set input video
    frame_width = int(cap.get(3))                                   # Returns the width and height of capture video
    frame_height = int(cap.get(4))
    # Set out for video writer
    out = cv2.VideoWriter(                                          # Set the Output path for video writer
        F"./Demo/{in1234}.avi", cv2.VideoWriter_fourcc(*"MJPG"), 25.0,
        (frame_width, frame_height))

    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(frame_width, frame_height, 3) # Create image according darknet for compatibility of network
    while not keyboard.is_pressed('q'):                                                     # Load the input frame and write output frame.
        prev_time = time.time()
        ret, frame_read = cap.read()                                 # Capture frame and return true if frame present
        # For Assertion Failed Error in OpenCV
        if not ret:                                                  # Check if frame present otherwise he break the while loop
            break

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)      # Convert frame into RGB from BGR and resize accordingly
        frame_resized = cv2.resize(frame_rgb,
                                   (frame_width, frame_height),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())                # Copy that frame bytes to darknet_image

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.4)    # Detection occurs at this line and return detections, for customize we can change the threshold.
        image = cvDrawBoxes(detections, frame_resized)               # Call the function cvDrawBoxes() for colored bounding box per class
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(1/(time.time()-prev_time))
        cv2.imshow('video detections', image)                                    # Display Image window
        cv2.waitKey(1)
        out.write(image)                                             # Write that frame into output video

        if len(detections)>0:
            for detection in detections:
                x, y, w, h = detection[2][0], \
                             detection[2][1], \
                             detection[2][2], \
                             detection[2][3]
                print(x, y, w, h)
                imgcroppped = image[int(y - (h / 2)):int(y + (h / 2)), int(x - (w / 2)):int(x + (w / 2))]
                # winname = "Cropped"
                # cv2.namedWindow(winname)
                # cv2.moveWindow(winname, 40, 30)
                # cv2.imshow(winname, imgcroppped)
                # cv2.waitKey(0)
                try:
                    imgcroppped1 = image[int(y - (h / 2) +10):int(y + (h / 2) -10), int(x - (w / 2) +10):int(x + (w / 2) -10)]
                    image = imgcroppped1
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray, (3, 3), 0)
                    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

                    # Morph open to remove noise and invert image
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
                    invert = 255 - opening

                    # Perform text extraction
                    data = pytesseract.image_to_string(invert, lang='eng', config='--psm 6')
                    print(data)

                    # cv2.imshow('thresh', thresh)
                    # cv2.imshow('opening', opening)
                    # cv2.imshow('invert', invert)
                    # cv2.waitKey()


                except Exception as e:        # w1 = 640
                    print(str(e))    # h1 = 480
                    imgcroppped = image[int(y - (h / 2)):int(y + (h / 2)), int(x - (w / 2)):int(x + (w / 2))]
                    image123 = imgcroppped
                    gray = cv2.cvtColor(image123, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray, (3, 3), 0)
                    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

                    # Morph open to remove noise and invert image
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
                    invert = 255 - opening

                    # Perform text extraction
                    data = pytesseract.image_to_string(invert, lang='eng', config='--psm 6')
                    print(data)

    cap.release()                                                    # For releasing cap and out.
    out.release()
    print(":::Video Write Completed")
    #exit()

if __name__ == "__main__":
    in123 = input("Enter name of mp4 video saved in data folder without its extension: ")
    in1234 = F'{in123}_output'
    YOLO(in123, in1234)                                                           # Calls the main function YOLO()
