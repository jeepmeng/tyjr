import cv2
import numpy as np

inpWidth = 640
inpHeight = 640
confThreshold = 0.3
nmsThreshold = 0.5
objThreshold = 0.2
def postprocess(Height,Width, outs):

    # frameHeight = frame.shape[0]
    # frameWidth = frame.shape[1]
    ratioh, ratiow = Height / inpWidth, Width / inpHeight
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.

    confidences = []
    boxes = []
    landmarks = []
    locations = []
    for detection in outs:
        confidence = detection[15]
        # if confidence > self.confThreshold and detection[4] > self.objThreshold:
        if detection[4] > objThreshold:
            center_x = int(detection[0] * ratiow)
            center_y = int(detection[1] * ratioh)
            width = int(detection[2] * ratiow)
            height = int(detection[3] * ratioh)
            left = int(center_x - width / 2)
            top = int(center_y - height / 2)

            confidences.append(float(confidence))
            # print(confidences)
            boxes.append([left, top, width, height])
            landmark = detection[5:15] * np.tile(np.float32([ratiow, ratioh]), 5)
            landmarks.append(landmark.astype(np.int32))
    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    # print(boxes)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    # print('indices=-=--=-=-=-=-=',indices)
    for i in indices:
        # i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        location = (top, left + width, top + height, left)
        locations.append(location)



    return locations