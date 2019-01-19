from objectDetector import ObjectDetector
import tensorflow as tf
import object_detection
from object_detection.utils import label_map_util

import cv2
import numpy as np
import os
import sys

model_path = "./data/models/knife_ssd/frozen_inference_graph.pb"

NUM_CLASSES = 1
# label_map = label_map_util.load_labelmap('/home/ruth/Documents/Bumblebee/ML/models/label_map.pbtxt')
label_map = label_map_util.load_labelmap('./data/labels/knife_label.pbtxt')

class KnifeDetector (ObjectDetector):
    def __init__ (self, min_score_thresh=.5):
        self.min_score_thresh = min_score_thresh
        self.load_model()


    def load_model(self):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        # load session
        self.sess = tf.Session(graph=self.detection_graph, config=tf.ConfigProto(allow_soft_placement=True))

        with self.detection_graph.as_default():

            self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            # self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.detection_boxes = tf.get_default_graph().get_tensor_by_name('detection_boxes:0')
            self.detection_scores = tf.get_default_graph().get_tensor_by_name('detection_scores:0')
            self.detection_classes = tf.get_default_graph().get_tensor_by_name('detection_classes:0')
            self.num_detections = tf.get_default_graph().get_tensor_by_name('num_detections:0')


    def detect(self, img, out_img=None):

        image_np_expanded = np.expand_dims(img, axis=0)

        output_dict = self.sess.run({ \
                            'detection_boxes': self.detection_boxes, \
                            'detection_scores': self.detection_scores, \
                            'detection_classes': self.detection_classes, \
                            'num_detections': self.num_detections},
                            feed_dict={self.image_tensor: image_np_expanded})

        if out_img is None:
            out_img = img.copy()

        return self.extract(out_img, output_dict)

    def extract(self, out_img, output_dict):
        height, width, _ = out_img.shape

        boxes = np.squeeze(output_dict['detection_boxes'])
        classes =  np.squeeze(output_dict['detection_classes']).astype(np.int32)
        scores = np.squeeze(output_dict['detection_scores'])

        objs = []

        boxes = boxes[scores > self.min_score_thresh]
        classes = classes[scores > self.min_score_thresh]
        scores = scores[scores > self.min_score_thresh]

        for i in range(boxes.shape[0]):
            # box = tuple(boxes[i].tolist())

            if scores[i] > self.min_score_thresh:
                box = tuple(boxes[i].tolist())
            else:
                continue

            display_str = ''
            if classes[i] in self.category_index.keys() and str(self.category_index[classes[i]]['name']) == 'knife':
                pass
            else:
                continue
        
            # ymin, xmin, ymax, xmax = boxes[i]
            # (left, right, top, bottom) = (xmin * width, xmax * width, \
            #                               ymin * height, ymax * height)

            objs.append([scores[i] ,boxes[i]])

        return objs
