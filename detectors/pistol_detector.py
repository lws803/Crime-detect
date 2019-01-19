import cv2
import os
import numpy as np
import sys
import tensorflow as tf
import timeit
import time
import math
import logging

from objects.humanDetector import HumanDetector

label_lines = [line.rstrip() for line
           in tf.gfile.GFile('./data/labels/gun_labels.txt')]

SCALE = 0.3
SCORE_THRESH = 0.7
TIME_THRESH = 300000 # 5 mins

logger = logging.getLogger("Pistol Detector")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

class PistolDetector:
    def __init__ (self, log_level=logging.DEBUG):
        logger.setLevel(log_level)
        self.initialSetup()
        self.sess = tf.Session()
        self.start_time = timeit.default_timer()

        self.softmax_tensor = self.sess.graph.get_tensor_by_name('final_result:0')
        logger.info('Took {} seconds to feed data to graph'.format(timeit.default_timer() - self.start_time))
        
        self.hd = HumanDetector(0.3)
        self.votes = []


    def initialSetup(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        start_time = timeit.default_timer()

        # This takes 2-5 seconds to run
        # Unpersists graph from file
        with tf.gfile.FastGFile('./data/models/gun_model/retrained_graph_gun.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    
        logger.info('Took {} seconds to unpersist the graph'.format(timeit.default_timer() - start_time))


    def detect(self, frame):
        if frame is None:
            raise SystemError('Issue grabbing the frame')
        
        frame = cv2.resize(frame,None,fx=SCALE,fy=SCALE)
        debugImage = frame.copy()

        humans = self.hd.detect(frame)

        # TODO: Do the cropping here
        height, width, channels = frame.shape

        if (len(self.votes) >= 1):
            if (time.time() - self.votes[-1] > TIME_THRESH):
                self.votes = []

        highestScore = 0

        for human in humans:
            humanRect = human
            humanBox = [0,0,0,0]

            humanBox[0] = int(humanRect[1]*width) # xmin
            humanBox[1] = int(humanRect[0]*height) # ymin
            humanBox[2] = int(humanRect[3]*width) # xmax
            humanBox[3] = int(humanRect[2]*height) # ymax
            h = (humanBox[3] - humanBox[1])/2
            h = int(h)
            crop_img = frame[humanBox[1]:humanBox[1]+h, humanBox[0]:humanBox[2]]
            cv2.rectangle(debugImage, (humanBox[0], humanBox[1]), (humanBox[2], humanBox[1]+h), (0, 255, 255), 1)

        #     # adhere to TS graph input structure
            crop_img = cv2.resize(crop_img, (299, 299), interpolation=cv2.INTER_CUBIC)

            numpy_frame = np.asarray(crop_img)
            numpy_frame = cv2.normalize(numpy_frame.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
            numpy_final = np.expand_dims(numpy_frame, axis=0)

            # This takes 2-5 seconds as well
            predictions = self.sess.run(self.softmax_tensor, {'Mul:0': numpy_final})
            
            # Sort to show labels of first prediction in order of confidence
            # TODO: Need tot test this on multiple persons, see if the classifier on its own can work on multiple persons
            # If not then we need to use human detector
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            for node_id in top_k:
                human_string = label_lines[node_id]
                if (human_string == "person handgun"):
                    score = predictions[0][node_id]
                    logger.info('%s (score = %.5f)' % (human_string, score))
                    # Get the highest prediction
                    highestScore = max([highestScore, score])
                    if (score > SCORE_THRESH):
                        self.votes.append(time.time())
                    break

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(debugImage, str(highestScore), (2,10), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)


        # frame = cv2.resize(frame, (299, 299), interpolation=cv2.INTER_CUBIC)         

        self.start_time = timeit.default_timer()

        logger.info('Took {} seconds to perform prediction'.format(timeit.default_timer() - self.start_time))

        self.start_time = timeit.default_timer()

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

    pd = PistolDetector()

    while(True):
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (400, 400))
        frame = pd.detect(frame)


        frame = cv2.resize(frame, (0,0), fx=1.0/SCALE, fy=1.0/SCALE) # Scale resizing

        cv2.imshow('image', frame)

        cv2.waitKey(1)
        # exit if the key "q" is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    