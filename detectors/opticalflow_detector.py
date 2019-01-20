import cv2
import sys
import numpy as np
import time
import math
import logging

from utils.common import *

from objects.cnnDetector import CNNDetector
from objects.humanDetector import HumanDetector

SCALE = 0.3
# maxAverage = -1
VOTE_THRESH = 5
PROBABILITY_THRESH = 0.6
TIME_THRESH = 300000 # 5 mins
GAMMA_VALUE = 2

logger = logging.getLogger("Optical Flow Detector")

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

class OpticalflowDetector:
    def __init__ (self, frame, log_level=logging.DEBUG):
        logger.setLevel(log_level)
        frame = cv2.resize(frame,None,fx=SCALE,fy=SCALE)
        frame = self.adjust_gamma(frame, gamma=GAMMA_VALUE)

        self.prevgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.fps_time = 0
        self.cnn = CNNDetector()
        self.humanDetector = HumanDetector(0.3)
        self.votes = []

    def adjust_gamma(self, image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
    
        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)


    def detect(self, frame):
        # global maxAverage
        frame = cv2.resize(frame,None,fx=SCALE,fy=SCALE)

        frame = self.adjust_gamma(frame, gamma=GAMMA_VALUE)

        debugImage = frame.copy()

        height, width, channels = frame.shape
        grayVelocity = np.zeros([height, width, 1], dtype=np.uint8)

        # Everything is phased out by one frame
        knifeBoxes = self.cnn.detect(frame)
        humans = self.humanDetector.detect(frame)

        # convert image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sample_rate = 3
        flow = cv2.calcOpticalFlowFarneback(self.prevgray, gray, None, 0.5, 3, 15, sample_rate, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

        grayVelocity[...,0] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX) 
        # Deprecated, we dont need the normalised grayscale

        # Erase entire array when latest data is stale
        if (len(self.votes) >= 1):
            if (time.time() - self.votes[-1] > TIME_THRESH):
                self.votes = []

        for human in humans:
            knifeConfidence = 0

            humanRect = human
            humanBox = [0,0,0,0]
            # Convert knife rect
            humanBox[0] = int(humanRect[1]*width) # xmin
            humanBox[1] = int(humanRect[0]*height) # ymin
            humanBox[2] = int(humanRect[3]*width) # xmax
            humanBox[3] = int(humanRect[2]*height) # ymax

            cv2.rectangle(debugImage, (humanBox[0], humanBox[1]), (humanBox[2], humanBox[3]), (0, 255, 255), 1)
            cv2.rectangle(grayVelocity, (humanBox[0], humanBox[1]), (humanBox[2], humanBox[3]), (255), 1)

            humanBoxArea = (humanBox[2] - humanBox[0])*(humanBox[3] - humanBox[1])

            for knife in knifeBoxes:
                knifeRect = knife[2]
                knifeBox = [0,0,0,0]
                # Convert knife rect
                knifeBox[0] = int(knifeRect[1]*width) # xmin
                knifeBox[1] = int(knifeRect[0]*height) # ymin
                knifeBox[2] = int(knifeRect[3]*width) # xmax
                knifeBox[3] = int(knifeRect[2]*height) # ymax
                cv2.rectangle(debugImage, (knifeBox[0], knifeBox[1]), (knifeBox[2], knifeBox[3]), (0, 0, 255), 2)

                if (isIntersect(humanBox, knifeBox)):
                    knifeConfidence = max(knifeConfidence, knife[1])

            croppedImage = mag[humanBox[0]:humanBox[2], humanBox[1]:humanBox[3]]
            average = croppedImage.mean(axis=0).mean(axis=0)

            # if average > maxAverage:
               # maxAverage = average

            if (math.isnan(average)):
                average = 0
            # Divide so that the closer the person is, the less likely he'll be giving off the false signal
            logger.info("=======================")
            logger.info('knife confidence intersecting with human: ' + str(knifeConfidence))
            logger.info('average motion within human: ' + str(average))
            # logger.info('maximum average motion within human: ' + str(maxAverage))

            # TODO: Add a probability function
            # Maximum average value seen: 7, will square the value for the probability function
            # Allocation to the probability: 0.5 to the average, 0.5 to the knife confidence
            pr = average/18 + (knifeConfidence)/2
            logger.info('combined pr: ' + str(pr))
            if (pr > PROBABILITY_THRESH):
                self.votes.append(time.time())

        # Update the previous
        self.prevgray = gray


        # For every human bounding box, find out if there is high activity within that area and if the knife overlaps it
        cv2.putText(debugImage, 
            "FPS: %f" % (1.0 / (time.time() - self.fps_time)),
            (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 0), 2)

        # cv2.imshow('Gray scale velocity',grayVelocity)
        draw_flow(debugImage, flow)

        self.fps_time = time.time()

        return debugImage

    def getVotes(self):
        # With this array, we can sort by timing, do range queries, and find total length of the votes
        return self.votes

    def clearVotes(self):
        self.votes = []


# main
if __name__ == '__main__':
    
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) # Scale resizing

    od = OpticalflowDetector(frame, log_level=logging.ERROR)

    while(True):
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) # Scale resizing
        frame = od.detect(frame)
        if (len(od.getVotes()) >= 5):
            print ("Notified")
            od.clearVotes()

        # frame = cv2.resize(frame, (0,0), fx=1.0/SCALE, fy=1.0/SCALE) # Scale resizing

        cv2.imshow('image', frame)

        cv2.waitKey(30)
        # exit if the key "q" is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
