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
import logging

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

class WorkerPistol(QtCore.QObject):
    finished = pyqtSignal() # our signal out to the main thread to alert it we've completed our work
    VideoSignal = QtCore.pyqtSignal(QtGui.QImage)
    loaded = pyqtSignal()

    def __init__(self):
        super(WorkerPistol, self).__init__()
        self.working = True # this is our flag to control our loop 
        self.flag = True

    def work(self):
        pd = PistolDetector(log_level=logging.DEBUG) 
        self.image_viewer_pistol = ImageViewerPistol()
        self.VideoSignal.connect(self.image_viewer_pistol.setImage)  

        while self.working:
            ret, frame = camera.read()
            frame = pd.detect(frame)
            frame = cv2.resize(frame, (400, 400))
            color_swapped_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, _ = color_swapped_image.shape

            votes = len(pd.getVotes()) # Get number of votes here

            qt_image = QtGui.QImage(color_swapped_image.data,
                                    width,
                                    height,
                                    color_swapped_image.strides[0],
                                    QtGui.QImage.Format_RGB888)

            self.VideoSignal.emit(qt_image)

            if (self.flag):
                self.loaded.emit()
                self.flag = False

            clear_output()

        self.flag = True
        # self.VideoSignal.emit(QtGui.QImage("white.jpg"))       
        qt_image = QtGui.QImage("offline-pistol.png")
        self.VideoSignal.emit(qt_image)
        self.finished.emit() # alert our gui that the loop stopped

class WorkerKnife(QtCore.QObject):
    finished = pyqtSignal() # our signal out to the main thread to alert it we've completed our work
    VideoSignal = QtCore.pyqtSignal(QtGui.QImage)
    loaded = pyqtSignal()

    def __init__(self):
        super(WorkerKnife, self).__init__()
        self.working = True # this is our flag to control our loop 
        self.flag = True

    def work(self):
        ret, frame = camera.read()
        od = OpticalflowDetector(frame, log_level=logging.DEBUG)
        self.VideoSignal.connect(self.image_viewer_knife.setImage)  

        while self.working:
            ret, frame = camera.read()
            debugImage = od.detect(frame)
            outputImage = cv2.resize(debugImage, (400, 400))

            color_swapped_image = cv2.cvtColor(outputImage, cv2.COLOR_BGR2RGB)

            height, width, _ = color_swapped_image.shape

            votes = len(od.getVotes()) # Get number of votes here

            qt_image = QtGui.QImage(color_swapped_image.data,
                                    width,
                                    height,
                                    color_swapped_image.strides[0],
                                    QtGui.QImage.Format_RGB888)

            self.VideoSignal.emit(qt_image)

            if (self.flag):
                self.loaded.emit()
                self.flag = False

            cv2.waitKey(40)
            clear_output()

        self.flag = True
        # self.VideoSignal.emit(QtGui.QImage("white.jpg"))       
        qt_image = QtGui.QImage("offline-knife.png")
        self.VideoSignal.emit(qt_image)
        self.finished.emit() # alert our gui that the loop stopped

class Window(QtWidgets.QMainWindow):

    def __init__(self):
        super(Window, self).__init__()
        self.setWindowTitle("Crime Predict")
        self.setStyleSheet("background-color: white;")         

        self.thread_pistol = None
        self.thread_knife = None
        self.worker_pistol = None
        self.worker_knife = None
   
        #Buttons to start the videocapture:
     
        self.push_button_pistol_start = QtWidgets.QPushButton('Start Pistol_Detect')
        self.push_button_pistol_start.clicked.connect(self.start_pistol_detect)
        self.push_button_pistol_stop = QtWidgets.QPushButton('Stop Pistol_Detect')
        self.push_button_pistol_stop.hide()

        self.push_button_knife_start = QtWidgets.QPushButton('Start Knife_Detect')
        self.push_button_knife_start.clicked.connect(self.start_knife_detect)
        self.push_button_knife_stop = QtWidgets.QPushButton('Stop Knife_Detect')
        self.push_button_knife_stop.hide()

        self.w1 = QtWidgets.QLabel()
        self.w1.setPixmap(QtGui.QPixmap("offline-pistol.png"))
        self.w2 = QtWidgets.QLabel()
        self.w2.setPixmap(QtGui.QPixmap("offline-knife.png"))

        # self.horizontal_layout_videos = QtWidgets.QHBoxLayout()
        # self.horizontal_layout_videos.addWidget(self.l1)
        # self.horizontal_layout_videos.addWidget(self.l2)

        self.pistol_buttons = QtWidgets.QVBoxLayout()
        self.pistol_buttons.addWidget(self.push_button_pistol_start)
        self.pistol_buttons.addWidget(self.push_button_pistol_stop)
        self.pistol_buttons.addWidget(self.w1)

        self.knife_buttons = QtWidgets.QVBoxLayout()
        self.knife_buttons.addWidget(self.push_button_knife_start)
        self.knife_buttons.addWidget(self.push_button_knife_stop)
        self.knife_buttons.addWidget(self.w2)

        self.horizontal_layout_buttons = QtWidgets.QHBoxLayout()
        self.horizontal_layout_buttons.addLayout(self.pistol_buttons)
        self.horizontal_layout_buttons.addLayout(self.knife_buttons)

        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.addLayout(self.horizontal_layout_buttons)    
        # self.main_layout.addLayout(self.horizontal_layout_videos)

        self.layout_widget = QtWidgets.QWidget()
        self.layout_widget.setLayout(self.main_layout)

        self.setCentralWidget(self.layout_widget)

    def remove_pistol_load(self):
        self.pistol_buttons.removeWidget(self.l1)
        self.pistol_buttons.addWidget(self.pistol_video)

    def remove_knife_load(self):
        self.knife_buttons.removeWidget(self.l2)
        self.knife_buttons.addWidget(self.knife_video)

    def start_pistol_detect(self):
        self.push_button_pistol_start.hide()
        self.push_button_pistol_stop.show()

        self.pistol_buttons.removeWidget(self.w1)
        self.l1 = QtWidgets.QLabel()
        loading = QtGui.QMovie("loading.gif")
        self.l1.setMovie(loading)
        loading.start()
        self.pistol_buttons.addWidget(self.l1)             

        self.thread_pistol = QtCore.QThread()  # a new thread to run our background tasks in
        self.worker_pistol = WorkerPistol()  # a new worker to perform those tasks
        self.worker_pistol.moveToThread(self.thread_pistol)  # move the worker into the thread, do this first before connecting the signals
        
        self.worker_pistol.image_viewer_pistol = ImageViewerPistol()
        self.worker_pistol.VideoSignal.connect(self.worker_pistol.image_viewer_pistol.setImage)

        self.pistol_video = QtWidgets.QWidget()
        self.pistol_video = self.worker_pistol.image_viewer_pistol

        self.thread_pistol.started.connect(self.worker_pistol.work)  # begin our worker object's loop when the thread starts running
        self.worker_pistol.loaded.connect(self.remove_pistol_load)
        self.push_button_pistol_stop.clicked.connect(self.stop_pistol)  # stop the loop on the stop button click
        self.worker_pistol.finished.connect(self.loop_finished)  # do something in the gui when the worker loop ends
        self.worker_pistol.finished.connect(self.thread_pistol.quit)  # tell the thread it's time to stop running
        self.worker_pistol.finished.connect(self.worker_pistol.deleteLater)  # have worker mark itself for deletion
        self.thread_pistol.finished.connect(self.thread_pistol.deleteLater)  # have thread mark itself for deletion
        # make sure those last two are connected to themselves or you will get random crashes

        self.thread_pistol.start()

    def start_knife_detect(self):
        self.push_button_knife_start.hide()
        self.push_button_knife_stop.show()

        self.knife_buttons.removeWidget(self.w2)
        self.l2 = QtWidgets.QLabel()
        loading = QtGui.QMovie("loading.gif")
        self.l2.setMovie(loading)
        loading.start()
        self.knife_buttons.addWidget(self.l2)            

        self.thread_knife = QtCore.QThread()  # a new thread to run our background tasks in
        self.worker_knife = WorkerKnife()  # a new worker to perform those tasks
        self.worker_knife.moveToThread(self.thread_knife)  # move the worker into the thread, do this first before connecting the signals

        self.worker_knife.image_viewer_knife = ImageViewerKnife()
        self.worker_knife.VideoSignal.connect(self.worker_knife.image_viewer_knife.setImage)

        self.knife_video = QtWidgets.QWidget()
        self.knife_video = self.worker_knife.image_viewer_knife

        # self.knife_buttons.removeWidget(self.l2)
        # self.knife_buttons.addWidget(self.knife_video)

        self.thread_knife.started.connect(self.worker_knife.work)  # begin our worker object's loop when the thread starts running
        self.worker_knife.loaded.connect(self.remove_knife_load)
        self.push_button_knife_stop.clicked.connect(self.stop_knife)  # stop the loop on the stop button click
        self.worker_knife.finished.connect(self.loop_finished)  # do something in the gui when the worker loop ends
        self.worker_knife.finished.connect(self.thread_knife.quit)  # tell the thread it's time to stop running
        self.worker_knife.finished.connect(self.worker_knife.deleteLater)  # have worker mark itself for deletion
        self.thread_knife.finished.connect(self.thread_knife.deleteLater)  # have thread mark itself for deletion
        # make sure those last two are connected to themselves or you will get random crashes

        self.thread_knife.start()

    def stop_pistol(self):
        self.worker_pistol.working = False
        self.pistol_buttons.removeWidget(self.pistol_video)
        self.push_button_pistol_start.show()
        self.push_button_pistol_stop.hide()
        self.pistol_buttons.addWidget(self.w1)

        # self.l1.setPixmap(QtGui.QPixmap("white.jpg"))
        # self.horizontal_layout_videos.addWidget(self.l1)

        # since thread's share the same memory, we read/write to variables of objects running in other threads
        # so when we are ready to stop the loop, just set the working flag to false

    def stop_knife(self):
        self.worker_knife.working = False
        self.knife_buttons.removeWidget(self.knife_video)
        self.push_button_knife_start.show()
        self.push_button_knife_stop.hide()
        self.knife_buttons.addWidget(self.w2)

        # self.l2.setPixmap(QtGui.QPixmap("white.jpg"))
        # self.horizontal_layout_videos.addWidget(self.l2)

    def loop_finished(self):
        # received a callback from the thread that it completed     
        print('Looped Finished')

def main():  
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 