import sys

# from src.insightface.src.common import face_preprocess
from src.insightface.src.common import face_preprocess

sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')

from mtcnn.mtcnn import MTCNN
# import face_preprocess
import numpy as np
import cv2
import os
from datetime import datetime


class TrainingDataCollector:

    def __init__(self, args):
        self.args = args
        # Detector = mtcnn_detector
        self.detector = MTCNN()

    def collectImagesFromCamera(self):
        # initialize video stream
        cap = cv2.VideoCapture(0)

        # Setup some useful var
        faces = 0
        frames = 0
        max_faces = int(self.args["faces"])
        max_bbox = np.zeros(4)

        if not (os.path.exists(self.args["output"])):
            os.makedirs(self.args["output"])

        while faces < max_faces:
            ret, frame = cap.read()
            frames += 1

            dtString = str(datetime.now().microsecond)
            # Get all faces on current frame
            bboxes = self.detector.detect_faces(frame) # return box,confidence, left_eye,right_eye,nose etc
            #print(bboxes)

            if len(bboxes) != 0:
                # Get only the biggest face
                max_area = 0
                for bboxe in bboxes:
                    bbox = bboxe["box"]
                    #print("bbox1",bbox)
                    bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]) # edited to get full face.
                    #print("bbox2",bbox)
                    keypoints = bboxe["keypoints"]
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if area > max_area:
                        max_bbox = bbox ## edited full face bboxs
                        #print("max_bbox",max_bbox)
                        landmarks = keypoints ## left_eye,right_eye,nose,mouth_left,mouth_right etc
                        # print("landmarks",landmarks)
                        max_area = area

                max_bbox = max_bbox[0:4]

                # get each of 3rd frames
                if frames % 3 == 0:
                    # convert to face_preprocess.preprocess input

                    # print([landmarks["left_eye"][0]])
                    # print([landmarks["left_eye"][1]])

                    landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0],
                                          landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                          landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1],
                                          landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
                    # print("landmarks_0,1",landmarks)

                    landmarks = landmarks.reshape((2, 5)).T
                    # landmark form :     [[229 238]
                    #                      [270 237]
                    #                      [260 258]
                    #                      [237 287]
                    #                      [268 286]]
                    # print("landmark_reshape", landmarks.reshape((2,5)))
                    # print("landmarks_T",landmarks)

                    nimg = face_preprocess.preprocess(frame, max_bbox, landmarks, image_size='112,112')

                    cv2.imwrite(os.path.join(self.args["output"], "{}.jpg".format(dtString)), nimg)
                    cv2.rectangle(frame, (max_bbox[0], max_bbox[1]), (max_bbox[2], max_bbox[3]), (255, 0, 0), 2)
                    print("[INFO] {} Image Captured".format(faces + 1))
                    faces += 1
            cv2.imshow("Face detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
