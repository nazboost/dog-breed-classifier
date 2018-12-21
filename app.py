"""
Detect dogs in real time and discriminate breeds.

Usage: python3 app.py [--video=path/to/video]
"""

import argparse
from datetime import datetime
import time
import os
import sys

import cv2
from keras.models import load_model
import numpy as np
from PIL import Image
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
import seaborn as sns

from model.model import DetectDogs
from search import search

INPUT_SIZE = 299

model = None
icon_path = './design/icon.png'
splash_pix_path = './design/splash_pix.jpg'
model_path = './model/classification/model_16.h5'


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        open_file_broeser_action = QtWidgets.QAction('&Open', self)
        open_file_broeser_action.setShortcut('Ctrl+O')
        open_file_broeser_action.triggered.connect(self.open_file_broeser)

        exit_action = QtWidgets.QAction('&Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(sys.exit)

        menubar = self.menuBar()
        file_menu = menubar.addMenu('&File')
        file_menu.addAction(open_file_broeser_action)
        file_menu.addAction(exit_action)

        self.statusBar().showMessage('Push SPACE to take photo and get description.')

    def open_file_broeser(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open File', '', "JPG Files (*.jpg);;All Files (*)", options=options)

        # TODO: I want to predict local file, but instance for detection is out
        # of scope.
        if file_name:
            frame_pil = Image.open(file_name)


class SubWindow(QtWidgets.QWidget):
    # TODO: Set window title.
    def __init__(self, parent=None, keyword=None):
        super(SubWindow, self).__init__(parent)

        self.sub_window = QtWidgets.QDialog(parent)

        # self.setWindowTitle('Dog Breed Details')

        text_browser = QtWidgets.QTextBrowser()
        if keyword:
            text_browser.setPlainText(search(keyword))
        else:
            text_browser.setPlainText('Keyword is not set.')

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(text_browser)
        self.sub_window.setLayout(layout)

    def show(self):
        self.sub_window.exec_()


class VideoCaptureView(QtWidgets.QGraphicsView, DetectDogs):
    repeat_interval = 33  # ms

    def __init__(self, parent=None, video_path=None):
        super(VideoCaptureView, self).__init__(parent)

        self.pixmap = None
        self.item = None
        self.rect_items = []

        self.breeds = []
        with open('./model/breeds_16.txt') as f:
            lines = f.readlines()
            for line in lines:
                self.breeds.append(line.replace('\n', ''))

        color_palette = np.array(sns.color_palette(n_colors=len(self.breeds)))
        self.colors = color_palette * 255

        if video_path:
            self.capture = cv2.VideoCapture(video_path)
        else:
            self.capture = cv2.VideoCapture(0)

        if not self.capture.isOpened():
            print('Failed in opening webcam.')
            sys.exit()

        # Initialize drawing canvas.
        self.scene = QtWidgets.QGraphicsScene()
        self.setScene(self.scene)

        self.set_video_image()

        # Update timer constantly.
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.set_video_image)
        self.timer.start(self.repeat_interval)

    def make_sub_window(self):
        # TODO: Corresponds when two or more dogs are detected.
        sub_window = SubWindow(keyword=self.breed_label)
        sub_window.show()

    def keyPressEvent(self, event):
        """ Override QtWidgets.QGraphicsView.keyPressEvent. """
        key = event.key()
        if key == QtCore.Qt.Key_Space:
            save_dir = 'save/'
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(
                save_dir + datetime.now().strftime('%Y%m%d%H%M%S') + '.jpg',
                self.tmp_frame)

            try:
                self.make_sub_window()
            except AttributeError:
                print('No dogs are detected.')

        super(VideoCaptureView, self).keyPressEvent(event)

    def process_image(self, frame, coordinates):
        """
        Process image.

        Draw rectangle and text.

        Args:
            frame (np.ndarray): BGR image
            coordinates (list): Coordinates of dogs

        Returns:
            frame (np.ndarray): BGR image after processing
        """

        for coordinate in coordinates:
            frame_crop = frame[coordinate[1]:coordinate[3],
                               coordinate[0]:coordinate[2]]

            frame_crop = cv2.resize(frame_crop, (INPUT_SIZE, INPUT_SIZE)) / 255

            frame_crop = np.reshape(frame_crop, (1, INPUT_SIZE, INPUT_SIZE, 3))
            prediction = model.predict(frame_crop)
            idx = np.argmax(prediction)
            self.breed_label = self.breeds[idx]

            color = tuple(map(int, self.colors[idx]))

            # Draw label.
            x = coordinate[0] + 10
            y = coordinate[1] - 10 if coordinate[1] - \
                20 > 20 else coordinate[1] + 20
            cv2.putText(
                img=frame,
                text=self.breed_label,
                org=(
                    x,
                    y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA)

            # Draw renctangle.
            frame = cv2.rectangle(frame,
                                  (coordinate[0], coordinate[1]),
                                  (coordinate[2], coordinate[3]),
                                  color,
                                  thickness=2)

        return frame

    def set_video_image(self):
        """
        Capture video image from web camera.
        """

        status, frame = self.capture.read()

        if not status:
            print('Could not read frame.')
            sys.exit()

        height, width, dim = frame.shape
        bytes_per_line = dim * width

        self.tmp_frame = frame

        # Detect dogs
        frame_pil = Image.fromarray(
            np.uint8(
                cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB)))

        coordinates = self.detect_image(frame_pil)

        frame = self.process_image(frame, coordinates)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.image = QtGui.QImage(frame.data, width, height,
                                  bytes_per_line, QtGui.QImage.Format_RGB888)

        if self.pixmap is None:  # Fist, make instance.
            self.pixmap = QtGui.QPixmap.fromImage(self.image)
            self.item = QtWidgets.QGraphicsPixmapItem(self.pixmap)
            self.scene.addItem(self.item)  # Arrange on canvas
        else:
            # Second or later, change settings.
            self.pixmap.convertFromImage(self.image)
            self.item.setPixmap(self.pixmap)


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument(
        '--video',
        default=False,
        type=str,
        help='path to video to use instead of web camera')

    FLAGS = parser.parse_args()
    if FLAGS.video:
        video_path = FLAGS.video
    else:
        video_path = None

    app = QtWidgets.QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    app.setWindowIcon(QtGui.QIcon(icon_path))

    main_window = MainWindow()
    main_window.setWindowTitle('Dog Breeds Classifier')

    # Show loading secreen.
    splash_pix = QtGui.QPixmap(splash_pix_path)
    splash = QtWidgets.QSplashScreen(
        splash_pix, QtCore.Qt.WindowStaysOnBottomHint)
    splash.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint |
                          QtCore.Qt.FramelessWindowHint)
    splash.setEnabled(False)

    progress_bar = QtWidgets.QProgressBar(splash)
    progress_bar.setMaximum(100)
    progress_bar.setGeometry(10, splash_pix.height() -
                             40, splash_pix.width() - 20, 20)

    splash.show()
    splash.showMessage(
        "<h2><font color='white' face='arial black'>Loading classification model...</font></h2>",
        QtCore.Qt.AlignTop | QtCore.Qt.AlignCenter,
        QtCore.Qt.black)

    for i in range(0, 51):
        progress_bar.setValue(i)
        t = time.time()
        while time.time() < t + 0.005:
            app.processEvents()

    global model
    model = load_model(model_path)

    splash.showMessage(
        "<h2><font color='white' face='arial black'>Loading yolo model...</font></h2>",
        QtCore.Qt.AlignTop | QtCore.Qt.AlignCenter,
        QtCore.Qt.black)

    for i in range(51, 100):
        progress_bar.setValue(i)
        t = time.time()
        while time.time() < t + 0.01:
            app.processEvents()

    video_capture_view = VideoCaptureView(video_path=video_path)

    for i in range(99, 101):
        progress_bar.setValue(i)
        t = time.time()
        while time.time() < t + 0.01:
            app.processEvents()

    main_window.setCentralWidget(video_capture_view)
    main_window.show()
    splash.finish(main_window)

    app.exec_()

    video_capture_view.capture.release()


if __name__ == "__main__":
    main()
