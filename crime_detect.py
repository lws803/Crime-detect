#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import cv2
import argparse
import os
import numpy as np
import sys
import tensorflow as tf
import timeit
import time
import math

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSignal

from IPython.display import clear_output

from detectors.opticalflow_detector import OpticalflowDetector
from detectors.pistol_detector import PistolDetector

camera_port = 0
camera = cv2.VideoCapture(camera_port)

def grabVideoFeed():
    grabbed, frame = camera.read()
    return frame if grabbed else None

class ShowVideoPistol(QtCore.QObject):
 
    #initiating the built in camera
    VideoSignal = QtCore.pyqtSignal(QtGui.QImage)
 
    def __init__(self, parent = None):
        super(ShowVideoPistol, self).__init__(parent)
        self.run_video = True
 
    @QtCore.pyqtSlot()

    def startVideo(self):
 
        self.run_video = True
        pd = PistolDetector()

        while self.run_video:
            ret, frame = camera.read()

            frame = pd.detect(frame)

            frame = cv2.resize(frame, (400, 400))

            color_swapped_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
     
            height, width, _ = color_swapped_image.shape
     
            qt_image = QtGui.QImage(color_swapped_image.data,
                                    width,
                                    height,
                                    color_swapped_image.strides[0],
                                    QtGui.QImage.Format_RGB888)

            self.VideoSignal.emit(qt_image)

            clear_output()
    
    def stopVideo(self):
        self.run_video = False


class ShowVideoKnife(QtCore.QObject):
 
    #initiating the built in camera
    VideoSignal = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, parent = None):
        self.run_video = True
        super(ShowVideoKnife, self).__init__(parent)
 
    @QtCore.pyqtSlot()

    def startVideo(self):
        self.run_video = True
        ret, frame = camera.read()
        od = OpticalflowDetector(frame)

        while self.run_video:
            ret, frame = camera.read()
            debugImage = od.detect(frame)
            outputImage = cv2.resize(debugImage, (400, 400))

            color_swapped_image = cv2.cvtColor(outputImage, cv2.COLOR_BGR2RGB)

            height, width, _ = color_swapped_image.shape

            qt_image = QtGui.QImage(color_swapped_image.data,
                                    width,
                                    height,
                                    color_swapped_image.strides[0],
                                    QtGui.QImage.Format_RGB888)

            self.VideoSignal.emit(qt_image)
          
            cv2.waitKey(40)
            clear_output()

    def stopVideo (self):
        self.run_video = False

class ImageViewerPistol(QtWidgets.QWidget):
    def __init__(self, parent = None):
        super(ImageViewerPistol, self).__init__(parent)
        self.image = QtGui.QImage()
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)
 
 
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0,0, self.image)
        self.image = QtGui.QImage()
 
    def initUI(self):
        self.setWindowTitle('Test')
 
    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        if image.isNull():
            print("Viewer Dropped frame!")
 
        self.image = image
        if image.size() != self.size():
            self.setFixedSize(image.size())
        self.update()
 
class ImageViewerKnife(QtWidgets.QWidget):
    def __init__(self, parent = None):
        super(ImageViewerKnife, self).__init__(parent)
        self.image = QtGui.QImage()
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)
 
 
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0,0, self.image)
        self.image = QtGui.QImage()
 
    def initUI(self):
        self.setWindowTitle('Test')
 
    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        if image.isNull():
            print("Viewer Dropped frame!")
 
        self.image = image
        if image.size() != self.size():
            self.setFixedSize(image.size())
        self.update()

def main():  
    app = QtWidgets.QApplication(sys.argv)

    thread = QtCore.QThread()
    thread.start()

    thread2 = QtCore.QThread()
    thread2.start()

    vid_pistol = ShowVideoPistol()
    vid_pistol.moveToThread(thread)

    vid_knife = ShowVideoKnife()
    vid_knife.moveToThread(thread2)    

    image_viewer_pistol = ImageViewerPistol()
    image_viewer_knife = ImageViewerKnife()
 
    vid_pistol.VideoSignal.connect(image_viewer_pistol.setImage)
    vid_knife.VideoSignal.connect(image_viewer_knife.setImage)
 
    #Buttons to start the videocapture:
 
    push_button_pistol_start = QtWidgets.QPushButton('Start Pistol_Detect')
    push_button_pistol_start.clicked.connect(vid_pistol.startVideo)

    push_button_pistol_stop = QtWidgets.QPushButton('Stop Pistol_Detect')
    push_button_pistol_stop.clicked.connect(vid_pistol.stopVideo)

    push_button_knife_start = QtWidgets.QPushButton('Start Knife_Detect')
    push_button_knife_start.clicked.connect(vid_knife.startVideo) 

    push_button_knife_stop = QtWidgets.QPushButton('Stop Knife_Detect')
    push_button_knife_stop.clicked.connect(vid_knife.stopVideo)       

    horizontal_layout_videos = QtWidgets.QHBoxLayout()
 
    horizontal_layout_videos.addWidget(image_viewer_pistol)
    horizontal_layout_videos.addWidget(image_viewer_knife)

    pistol_buttons = QtWidgets.QVBoxLayout()
    pistol_buttons.addWidget(push_button_pistol_start)
    pistol_buttons.addWidget(push_button_pistol_stop)

    knife_buttons = QtWidgets.QVBoxLayout()
    knife_buttons.addWidget(push_button_knife_start)
    knife_buttons.addWidget(push_button_knife_stop)

    horizontal_layout_buttons = QtWidgets.QHBoxLayout()
    horizontal_layout_buttons.addLayout(pistol_buttons)
    horizontal_layout_buttons.addLayout(knife_buttons)

    main_layout = QtWidgets.QVBoxLayout()
    main_layout.addLayout(horizontal_layout_buttons)    
    main_layout.addLayout(horizontal_layout_videos)

    layout_widget = QtWidgets.QWidget()
    layout_widget.setLayout(main_layout)

    main_window = QtWidgets.QMainWindow()
    main_window.setCentralWidget(layout_widget)
    main_window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()   