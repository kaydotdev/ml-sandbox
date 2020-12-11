import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QHBoxLayout, QMessageBox, QGraphicsScene
from PyQt5.QtGui import QCursor, QImage, QPixmap

from infrastructure.model import NeuralNetwork


def rgb2gray(rgb):
    return np.dot(rgb[:, :, :3], [0.2989, 0.5870, 0.1140])


def parse_image(image: np.ndarray):
    assert (np.max(image) <= 255)
    image8 = image.astype(np.uint8, order='C', casting='unsafe')
    height, width, colors = image8.shape
    bytesPerLine = 3 * width

    image = QImage(image8.data,
                   width,
                   height,
                   bytesPerLine,
                   QImage.Format_RGB888)

    image = image.rgbSwapped()
    return image


class Window(QWidget):
    __canvas_size__ = QPoint(400, 400)

    def __init__(self):
        super().__init__()
        self.classification = NeuralNetwork()
        self.cursor = QCursor()
        self.setWindowTitle("MNIST UI")
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)
        self.setFixedSize(400, 460)
        self.canvas = np.zeros([self.__canvas_size__.x(), self.__canvas_size__.y(), 3])

        guess_btn = QPushButton(self)
        guess_btn.setText('Let me guess')
        guess_btn.move(210, 410)
        guess_btn.resize(180, 40)
        guess_btn.show()
        guess_btn.clicked.connect(self.alert_prediction)

        clear_btn = QPushButton(self)
        clear_btn.setText('Clear canvas')
        clear_btn.move(10, 410)
        clear_btn.resize(180, 40)
        clear_btn.show()
        clear_btn.clicked.connect(self.clear_board)

        self.picture = QLabel("", self)
        self.picture.resize(self.__canvas_size__.x(), self.__canvas_size__.y())
        self.picture.move(0, 0)
        self.picture.setStyleSheet("background-color:white;")
        self.draw_on_canvas()

        self.picture.show()
        self.show()

    def mouseMoveEvent(self, event):
        x = event.x()
        y = event.y()

        if event.buttons() == Qt.LeftButton:
            pen_radius = 10
            if pen_radius <= x < self.__canvas_size__.x() - pen_radius and \
               pen_radius <= y < self.__canvas_size__.y() - pen_radius:
                self.draw_dot(x, y, pen_radius)
                self.draw_on_canvas()

    def alert_prediction(self):
        alert = QMessageBox()
        alert.setWindowTitle("My guess is:")
        alert.setStyleSheet("QLabel{min-width: 300px;min-height: 50px;}")
        alert.setStandardButtons(QMessageBox.Ok)
        image = Image.fromarray(self.canvas.astype('uint8'), 'RGB').resize((28, 28))
        resized_picture = rgb2gray(np.array(image))
        result = self.classification.predict(resized_picture)
        alert.setText("The number is: {}".format(result))
        alert.exec_()

    def draw_on_canvas(self):
        self.picture.setPixmap(QPixmap.fromImage(parse_image(self.canvas)))

    def clear_board(self):
        self.canvas = np.zeros([self.__canvas_size__.x(), self.__canvas_size__.y(), 3])
        self.picture.setPixmap(QPixmap.fromImage(parse_image(self.canvas)))

    def draw_dot(self, x, y, size):
        for i in range(x - size, x + size):
            for j in range(y - size, y + size):
                self.canvas[j, i, 0] = 255
                self.canvas[j, i, 1] = 255
                self.canvas[j, i, 2] = 255
