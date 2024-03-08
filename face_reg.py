import cv2
import argparse
import numpy as np
import face_recognition
import cv2
import numpy as np
import os
import argparse

# known_face_encodings = []
# known_face_names = []
# img_file = os.listdir('./dataset')[1:]
#
# # print(img_file)
# for i in img_file:
#     name = i.split(".")[0]
#     name_img = cv2.imread(os.path.join('./dataset', i))
#     name_img = cv2.cvtColor(name_img, cv2.COLOR_BGR2RGB)
#     name_img = cv2.resize(name_img, (720, 1280), interpolation=cv2.INTER_LINEAR)
#     # print(name_img.shape)
#     name_face_encoding = face_recognition.face_encodings(name_img)[0]
#     known_face_encodings.append(name_face_encoding)
#     known_face_names.append(name)


# print('known_face_names---------------',known_face_names)


class yolov5():
    def __init__(self, yolo_type='yolov5s', confThreshold=0.3, nmsThreshold=0.5, objThreshold=0.2):
        anchors = [[4, 5, 8, 10, 13, 16], [23, 29, 43, 55, 73, 105], [146, 217, 231, 300, 335, 433]]
        num_classes = 1
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.no = num_classes + 5 + 10
        self.grid = [np.zeros(1)] * self.nl
        self.stride = np.array([8., 16., 32.])
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, -1, 2)
        self.inpWidth = 640
        self.inpHeight = 640
        self.net = cv2.dnn.readNet(yolo_type + '-face.onnx')
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        ratioh, ratiow = frameHeight / self.inpHeight, frameWidth / self.inpWidth
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.

        confidences = []
        boxes = []
        landmarks = []
        locations = []
        for detection in outs:
            confidence = detection[15]
            # if confidence > self.confThreshold and detection[4] > self.objThreshold:
            if detection[4] > self.objThreshold:
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
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
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

            # landmark = landmarks[i]
            # frame = self.drawPred(frame, locations, left, top, left + width, top + height, landmark)


        # frame_1 = self.drawPred(frame, locations)



        # print('locations=-=-=-=-=',locations)

        return locations
        # return frame_1

    def drawPred(self, frame_2, locations):
        # Draw a bounding box.

        face_encodings = face_recognition.face_encodings(frame_2, locations)
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.45)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)



        # print('face_names-=-=-=-=-=-=-=-',face_names)
        for (top, right, bottom, left), name in zip(locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            # top *= 4
            # right *= 4
            # bottom *= 4
            # left *= 4

            # Draw a box around the face
            cv2.rectangle(frame_2, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame_2, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame_2, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)
        # # label = '%.2f' % conf
        # # Display the label at the top of the bounding box
        # # labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # # top = max(top, labelSize[1])
        # # cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        # # for i in range(5):
        # #     cv2.circle(frame, (landmark[i*2], landmark[i*2+1]), 1, (0,255,0), thickness=-1)
        # # cv2.rectangle(frame, (left, top - 35), (right, top), (0, 0, 255), cv2.FILLED)
        # font = cv2.FONT_HERSHEY_DUPLEX
        # cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)
        return frame_2

    def detect(self, srcimg):
        blob = cv2.dnn.blobFromImage(srcimg, 1 / 255.0, (self.inpWidth, self.inpHeight), [0, 0, 0], swapRB=True,
                                     crop=False)
        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())[0]

        # inference output
        outs[..., [0, 1, 2, 3, 4, 15]] = 1 / (1 + np.exp(-outs[..., [0, 1, 2, 3, 4, 15]]))  ###sigmoid
        row_ind = 0
        for i in range(self.nl):
            h, w = int(self.inpHeight / self.stride[i]), int(self.inpWidth / self.stride[i])
            length = int(self.na * h * w)
            if self.grid[i].shape[2:4] != (h, w):
                self.grid[i] = self._make_grid(w, h)

            g_i = np.tile(self.grid[i], (self.na, 1))
            a_g_i = np.repeat(self.anchor_grid[i], h * w, axis=0)
            outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + g_i) * int(
                self.stride[i])
            outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * a_g_i

            outs[row_ind:row_ind + length, 5:7] = outs[row_ind:row_ind + length, 5:7] * a_g_i + g_i * int(
                self.stride[i])  # landmark x1 y1
            outs[row_ind:row_ind + length, 7:9] = outs[row_ind:row_ind + length, 7:9] * a_g_i + g_i * int(
                self.stride[i])  # landmark x2 y2
            outs[row_ind:row_ind + length, 9:11] = outs[row_ind:row_ind + length, 9:11] * a_g_i + g_i * int(
                self.stride[i])  # landmark x3 y3
            outs[row_ind:row_ind + length, 11:13] = outs[row_ind:row_ind + length, 11:13] * a_g_i + g_i * int(
                self.stride[i])  # landmark x4 y4
            outs[row_ind:row_ind + length, 13:15] = outs[row_ind:row_ind + length, 13:15] * a_g_i + g_i * int(
                self.stride[i])  # landmark x5 y5
            row_ind += length
        return outs


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--yolo_type', type=str, default='yolov5m', choices=['yolov5s', 'yolov5m', 'yolov5l'], help="yolo type")
    # parser.add_argument("--imgpath", type=str, default='selfie.jpg', help="image path")
    # parser.add_argument('--confThreshold', default=0.3, type=float, help='class confidence')
    # parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
    # parser.add_argument('--objThreshold', default=0.3, type=float, help='object confidence')
    # args = parser.parse_args()

    # video_capture = cv2.VideoCapture(0)

    # yolonet = yolov5(args.yolo_type, confThreshold=args.confThreshold, nmsThreshold=args.nmsThreshold, objThreshold=args.objThreshold)
    yolonet = yolov5()

    while True:
        ret, frame = video_capture.read()
        # srcimg = cv2.imread(frame)
        # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
        dets = yolonet.detect(frame)

        # print(dets)
        # print(dets.shape)
        # srcimg = yolonet.postprocess(frame, dets)
        srcimg = yolonet.postprocess(frame, dets)

        # winName = 'Deep learning object detection in OpenCV'
        # cv2.namedWindow(winName, 0)
        # for (top, right, bottom, left), name in zip(locations, facename):
        #     # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        #     top *= 4
        #     right *= 4
        #     bottom *= 4
        #     left *= 4
        #
        #     # Draw a box around the face
        #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        #
        #     # Draw a label with a name below the face
        #     cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        #     font = cv2.FONT_HERSHEY_DUPLEX
        #     cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        winName = 'Deep learning object detection in OpenCV'
        cv2.namedWindow(winName, 0)
        cv2.imshow(winName, srcimg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # cv2.waitKey(0)
    video_capture.release()
    cv2.destroyAllWindows()