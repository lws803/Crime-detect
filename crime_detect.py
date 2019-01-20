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
import datetime

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtMultimedia

from IPython.display import clear_output

from detectors.opticalflow_detector import OpticalflowDetector
from detectors.pistol_detector import PistolDetector

camera_port = 0
camera = cv2.VideoCapture(camera_port)

pics = ["white.jpg"]*12
SIZE = 0

def grabVideoFeed():
    grabbed, frame = camera.read()
    return frame if grabbed else None

class ImageViewer(QtWidgets.QWidget):
    def __init__(self, parent = None):
        super(ImageViewer, self).__init__(parent)
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
    finished = QtCore.pyqtSignal() # our signal out to the main thread to alert it we've completed our work
    VideoSignal = QtCore.pyqtSignal(QtGui.QImage)
    loaded = QtCore.pyqtSignal()
    alert = QtCore.pyqtSignal()

    def __init__(self):
        super(WorkerPistol, self).__init__()
        self.working = True # this is our flag to control our loop 
        self.flag = True
        self.alert_flag = True

    def work(self):
        pd = PistolDetector(log_level=logging.DEBUG) 
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

            if (votes >= 5) & (self.alert_flag):
                self.alert.emit()
                global SIZE
                pics[SIZE] = qt_image
                SIZE += 1
                SIZE = SIZE%12
                self.alert_flag = False

            clear_output()

        self.flag = True
        # self.VideoSignal.emit(QtGui.QImage("white.jpg"))       
        qt_image = QtGui.QImage("gun.jpeg")
        self.VideoSignal.emit(qt_image)
        self.finished.emit() # alert our gui that the loop stopped

class WorkerKnife(QtCore.QObject):
    finished = QtCore.pyqtSignal() # our signal out to the main thread to alert it we've completed our work
    VideoSignal = QtCore.pyqtSignal(QtGui.QImage)
    loaded = QtCore.pyqtSignal()
    alert = QtCore.pyqtSignal()

    def __init__(self):
        super(WorkerKnife, self).__init__()
        self.working = True # this is our flag to control our loop 
        self.flag = True
        self.alert_flag = True

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

            if (votes >= 5) & (self.alert_flag):
                self.alert.emit()
                frame = cv2.resize(frame, (400, 400))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, _ = frame.shape                
                capture = QtGui.QImage(frame.data,
                                    width,
                                    height,
                                    color_swapped_image.strides[0],
                                    QtGui.QImage.Format_RGB888)
                global SIZE
                pics[SIZE] = capture
                SIZE += 1
                SIZE = SIZE%12
                self.alert_flag = False

            cv2.waitKey(40)
            clear_output()

        self.flag = True
        # self.VideoSignal.emit(QtGui.QImage("white.jpg"))       
        qt_image = QtGui.QImage("knife.jpg")
        self.VideoSignal.emit(qt_image)
        self.finished.emit() # alert our gui that the loop stopped

class QtHandler(logging.Handler):
    def __init__(self):
        logging.Handler.__init__(self)
    def emit(self, record):
        record = self.format(record)
        if record: XStream.stdout().write('%s\n'%record)

logger_msg = logging.getLogger(__name__)
handler_msg = QtHandler()
handler_msg.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger_msg.addHandler(handler_msg)
logger_msg.setLevel(logging.DEBUG)

class XStream(QtCore.QObject):
    _stdout = None
    # _stderr = None
    messageWritten = QtCore.pyqtSignal(str)
    def flush( self ):
        pass
    def fileno( self ):
        return -1
    def write( self, msg ):
        if ( not self.signalsBlocked() ):
            self.messageWritten.emit(str(msg))
    @staticmethod
    def stdout():
        if ( not XStream._stdout ):
            XStream._stdout = XStream()
        return XStream._stdout


class ImagePopup(QtWidgets.QLabel):
    """ 
    The ImagePopup class is a QLabel that displays a popup, zoomed image 
    on top of another label.  
    """
    def __init__(self, parent):
        super(QtWidgets.QLabel, self).__init__(parent)
        
        thumb = parent.pixmap()
        imageSize = thumb.size()
        imageSize.setWidth(imageSize.width()*2)
        imageSize.setHeight(imageSize.height()*2)
        self.setPixmap(thumb.scaled(imageSize,QtCore.Qt.KeepAspectRatioByExpanding))
        
        # center the zoomed image on the thumb
        position = self.cursor().pos()
        position.setX(position.x() - thumb.size().width())
        position.setY(position.y() - thumb.size().height())
        self.move(position)
        
        # FramelessWindowHint may not work on some window managers on Linux
        # so I force also the flag X11BypassWindowManagerHint
        self.setWindowFlags(QtCore.Qt.Popup | QtCore.Qt.WindowStaysOnTopHint 
                            | QtCore.Qt.FramelessWindowHint 
                            | QtCore.Qt.X11BypassWindowManagerHint)

    def leaveEvent(self, event):
        """ When the mouse leave this widget, destroy it. """
        self.destroy()
        

class ImageLabel(QtWidgets.QLabel):
    """ This widget displays an ImagePopup when the mouse enter its region """
    def enterEvent(self, event):
        self.p = ImagePopup(self)
        self.p.show()
        event.accept() 

class ImageGallery(QtWidgets.QDialog):
    
    def __init__(self, parent=None):
        super(QtWidgets.QDialog, self).__init__(parent)
        self.setWindowTitle("Image Gallery")
        self.setLayout(QtWidgets.QGridLayout(self))
    
    def populate(self, pics, size, imagesPerRow=4, 
                 flags=QtCore.Qt.KeepAspectRatioByExpanding):
        row = col = 0
        for pic in pics:
            label = ImageLabel("")
            pixmap = QtGui.QPixmap(pic)
            pixmap = pixmap.scaled(size, flags)
            label.setPixmap(pixmap)
            self.layout().addWidget(label, row, col)
            col +=1
            if col % imagesPerRow == 0:
                row += 1
                col = 0

class Window(QtWidgets.QMainWindow):

    def __init__(self):
        super(Window, self).__init__()
        self.setWindowTitle("Crime Predict")
        self.setStyleSheet("background-color: white;")         

        self.thread_pistol = None
        self.thread_knife = None
        self.worker_pistol = None
        self.worker_knife = None

        # Setting up pistol buttons
        self.push_button_pistol_start = QtWidgets.QPushButton('Start Pistol_Detect')
        self.push_button_pistol_start.clicked.connect(self.start_pistol_detect)
        self.push_button_pistol_wait = QtWidgets.QPushButton('Please wait... Pistol_Detect is loading...')
        self.push_button_pistol_stop = QtWidgets.QPushButton('Stop Pistol_Detect')
        self.push_button_pistol_wait.hide()
        self.push_button_pistol_stop.hide()

        # Setting up knife buttons
        self.push_button_knife_start = QtWidgets.QPushButton('Start Knife_Detect')
        self.push_button_knife_start.clicked.connect(self.start_knife_detect)
        self.push_button_knife_wait = QtWidgets.QPushButton('Please Wait... Knife_Detect is loading...')
        self.push_button_knife_stop = QtWidgets.QPushButton('Stop Knife_Detect')
        self.push_button_knife_wait.hide() 
        self.push_button_knife_stop.hide()

        # Show initialised cam feed (offline)
        self.w1 = QtWidgets.QLabel()
        self.w1.setPixmap(QtGui.QPixmap("gun.jpeg"))
        self.w2 = QtWidgets.QLabel()
        self.w2.setPixmap(QtGui.QPixmap("knife.jpg"))

        # Vertical layout for pistol-related widgets
        self.pistol_buttons = QtWidgets.QVBoxLayout()
        self.pistol_buttons.addWidget(self.push_button_pistol_start)
        self.pistol_buttons.addWidget(self.push_button_pistol_wait)
        self.pistol_buttons.addWidget(self.push_button_pistol_stop)
        self.pistol_buttons.addWidget(self.w1)

        # Vertical layout for knife-related widgets
        self.knife_buttons = QtWidgets.QVBoxLayout()
        self.knife_buttons.addWidget(self.push_button_knife_start)
        self.knife_buttons.addWidget(self.push_button_knife_wait)
        self.knife_buttons.addWidget(self.push_button_knife_stop)
        self.knife_buttons.addWidget(self.w2)

        # Horizontal layout for both pistol and knife vertical layouts
        self.horizontal_layout_buttons = QtWidgets.QHBoxLayout()
        self.horizontal_layout_buttons.addLayout(self.pistol_buttons)
        self.horizontal_layout_buttons.addLayout(self.knife_buttons)

        # Text Browser Box for messages
        self.message_box = QtWidgets.QTextBrowser()
        self.message_box.setFixedSize(800,200)
        self.message_box.setFontFamily("Tahoma")
        XStream.stdout().messageWritten.connect( self.message_box.insertPlainText )
        self.message_box.insertPlainText('WELCOME TO CRIME_PREDICT\n')

        # Initialize tab screen
        self.tabs = QtWidgets.QTabWidget()
        self.tab1 = QtWidgets.QWidget()
        self.tab2 = QtWidgets.QWidget()
        
        # Add tabs
        self.tabs.addTab(self.tab1,"Camera Feeds")
        self.tabs.addTab(self.tab2,"Alerts and Events Center")               

        # Set Tab 1 layout
        self.tab1.layout = QtWidgets.QVBoxLayout()
        self.tab1.layout.addLayout(self.horizontal_layout_buttons)  
        self.tab1.layout.addWidget(self.message_box) 
        self.tab1.setLayout(self.tab1.layout)     

        # Image gallery
        self.ig = ImageGallery()
        self.ig.populate(pics, QtCore.QSize(200,200))
        self.ig.show()

        # Set Tab 2 layout    
        self.tab2.layout = QtWidgets.QVBoxLayout()
        # self.tab2.layout.addLayout(self.horizontal_layout_buttons)  
        self.tab2.layout.addWidget(self.ig) 
        self.tab2.setLayout(self.tab2.layout)   

        # Play start-up sound
        self.start_up_sound = QtMultimedia.QSoundEffect()
        self.start_up_sound.setSource(QtCore.QUrl.fromLocalFile('start-up.wav'))  
        self.start_up_sound.play()

        # Initialise sounds FX
        self.info_start = QtMultimedia.QSoundEffect()
        self.info_start.setSource(QtCore.QUrl.fromLocalFile('info-start.wav'))
        self.info_stop = QtMultimedia.QSoundEffect()
        self.info_stop.setSource(QtCore.QUrl.fromLocalFile('info-stop.wav'))     

        self.setCentralWidget(self.tabs)

    def remove_pistol_load(self):
        self.pistol_buttons.removeWidget(self.l1)
        self.pistol_buttons.addWidget(self.pistol_video)
        self.push_button_pistol_wait.hide()
        self.push_button_pistol_stop.show()

    def remove_knife_load(self):
        self.knife_buttons.removeWidget(self.l2)
        self.knife_buttons.addWidget(self.knife_video)
        self.push_button_knife_wait.hide()
        self.push_button_knife_stop.show()

    def alert_pistol(self):
        self.alert_box = QtWidgets.QMessageBox()
        self.alert_box.setIcon(QtWidgets.QMessageBox.Critical)
        self.alert_box.setText("Pistol Detected!")
        self.alert_box.show()
        self.alert_sound = QtMultimedia.QSoundEffect()
        self.alert_sound.setSource(QtCore.QUrl.fromLocalFile('siren.wav'))  
        self.alert_sound.play()
        self.ig.populate(pics, QtCore.QSize(200,200))        
        logger_msg.warning("Suspicious activity involving firearms detected @ " + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    def alert_knife(self):
        self.alert_box = QtWidgets.QMessageBox()
        self.alert_box.setIcon(QtWidgets.QMessageBox.Critical)
        self.alert_box.setText("Knife Detected!")
        self.alert_box.show()
        self.alert_sound = QtMultimedia.QSoundEffect()
        self.alert_sound.setSource(QtCore.QUrl.fromLocalFile('siren.wav'))  
        self.alert_sound.play()       
        self.ig.populate(pics, QtCore.QSize(200,200))         
        logger_msg.warning("Suspicious activity involving knives detected @ " + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    def start_pistol_detect(self):
        logger_msg.info("Pistol_Detect started")
        self.info_start.play()        
        self.push_button_pistol_start.hide()
        self.push_button_pistol_wait.show()

        self.pistol_buttons.removeWidget(self.w1)
        self.l1 = QtWidgets.QLabel()
        loading = QtGui.QMovie("loading.gif")
        self.l1.setMovie(loading)
        loading.start()
        self.pistol_buttons.addWidget(self.l1)             

        self.thread_pistol = QtCore.QThread()  # a new thread to run our background tasks in
        self.worker_pistol = WorkerPistol()  # a new worker to perform those tasks
        self.worker_pistol.moveToThread(self.thread_pistol)  # move the worker into the thread, do this first before connecting the signals
        
        self.worker_pistol.image_viewer_pistol = ImageViewer()
        self.worker_pistol.VideoSignal.connect(self.worker_pistol.image_viewer_pistol.setImage)

        self.pistol_video = QtWidgets.QWidget()
        self.pistol_video = self.worker_pistol.image_viewer_pistol

        self.thread_pistol.started.connect(self.worker_pistol.work)  # begin our worker object's loop when the thread starts running
        self.worker_pistol.loaded.connect(self.remove_pistol_load)
        self.worker_pistol.alert.connect(self.alert_pistol)
        self.push_button_pistol_stop.clicked.connect(self.stop_pistol)  # stop the loop on the stop button click
        self.worker_pistol.finished.connect(self.loop_finished)  # do something in the gui when the worker loop ends
        self.worker_pistol.finished.connect(self.thread_pistol.quit)  # tell the thread it's time to stop running
        self.worker_pistol.finished.connect(self.worker_pistol.deleteLater)  # have worker mark itself for deletion
        self.thread_pistol.finished.connect(self.thread_pistol.deleteLater)  # have thread mark itself for deletion
        # make sure those last two are connected to themselves or you will get random crashes

        self.thread_pistol.start()

    def start_knife_detect(self):
        logger_msg.info("Knife_Detect started")
        self.info_start.play()            
        self.push_button_knife_start.hide()
        self.push_button_knife_wait.show()

        self.knife_buttons.removeWidget(self.w2)
        self.l2 = QtWidgets.QLabel()
        loading = QtGui.QMovie("loading.gif")
        self.l2.setMovie(loading)
        loading.start()
        self.knife_buttons.addWidget(self.l2)            

        self.thread_knife = QtCore.QThread()  # a new thread to run our background tasks in
        self.worker_knife = WorkerKnife()  # a new worker to perform those tasks
        self.worker_knife.moveToThread(self.thread_knife)  # move the worker into the thread, do this first before connecting the signals

        self.worker_knife.image_viewer_knife = ImageViewer()
        self.worker_knife.VideoSignal.connect(self.worker_knife.image_viewer_knife.setImage)

        self.knife_video = QtWidgets.QWidget()
        self.knife_video = self.worker_knife.image_viewer_knife

        self.thread_knife.started.connect(self.worker_knife.work)  # begin our worker object's loop when the thread starts running
        self.worker_knife.loaded.connect(self.remove_knife_load)
        self.worker_knife.alert.connect(self.alert_knife)
        self.push_button_knife_stop.clicked.connect(self.stop_knife)  # stop the loop on the stop button click
        self.worker_knife.finished.connect(self.loop_finished)  # do something in the gui when the worker loop ends
        self.worker_knife.finished.connect(self.thread_knife.quit)  # tell the thread it's time to stop running
        self.worker_knife.finished.connect(self.worker_knife.deleteLater)  # have worker mark itself for deletion
        self.thread_knife.finished.connect(self.thread_knife.deleteLater)  # have thread mark itself for deletion
        # make sure those last two are connected to themselves or you will get random crashes

        self.thread_knife.start()

    def stop_pistol(self):
        self.worker_pistol.working = False
        self.info_stop.play()
        self.pistol_buttons.removeWidget(self.pistol_video)
        self.push_button_pistol_start.show()
        self.push_button_pistol_stop.hide()
        self.pistol_buttons.addWidget(self.w1)
        logger_msg.info("Pistol_Detect stopped")        

        # self.l1.setPixmap(QtGui.QPixmap("white.jpg"))
        # self.horizontal_layout_videos.addWidget(self.l1)

        # since thread's share the same memory, we read/write to variables of objects running in other threads
        # so when we are ready to stop the loop, just set the working flag to false

    def stop_knife(self):
        self.worker_knife.working = False
        self.info_stop.play()
        self.knife_buttons.removeWidget(self.knife_video)
        self.push_button_knife_start.show()
        self.push_button_knife_stop.hide()
        self.knife_buttons.addWidget(self.w2)
        logger_msg.info("Knife_Detect stopped")        

        # self.l2.setPixmap(QtGui.QPixmap("white.jpg"))
        # self.horizontal_layout_videos.addWidget(self.l2)

    def loop_finished(self):
        # received a callback from the thread that it completed     
        print('Looped Finished')


def main():  
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon('stop.jpg'))
    window = Window()
    window.show()

    # ig = ImageGallery()
    # ig.populate(pics, QtCore.QSize(200,200))
    # ig.show()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()    