"""
Detect dogs with yolov3 and classify breeds.
"""

import colorsys
import os

import cv2
from keras import backend as K
from keras import preprocessing
from keras.layers import Input
from keras.models import load_model
from keras.utils import multi_gpu_model
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFont, ImageDraw

from .yolo.yolo import Yolo
from .yolo.yolo_model import yolo_eval, yolo_body, tiny_yolo_body
from .yolo.yolo_utils import letterbox_image

INPUT_SIZE = 299


class DetectDogs(Yolo):
    def __init__(self):
        super().__init__()
        self.target_classes = ['dog']

    def detect_image(self, image):
        """
        Detect dogs.

        Args:
            image (pil):
                Target image

        Returns:
            coordinates (list):
                Each row has cordinates of rentangle.
        """

        # Over ride of parent's method
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required.'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required.'
            boxed_image = letterbox_image(
                image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found', len(out_boxes), 'objects by yolo.')

        coordinates = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]

            if predicted_class in self.target_classes:
                print('> Found', predicted_class, end=' ')

                box = out_boxes[i]
                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(
                    bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(
                    right + 0.5).astype('int32'))
                print((left, top), (right, bottom))

                coordinates.append([left, top, right, bottom])

        return coordinates


if __name__ == "__main__":
    detect_dogs = DetectDogs()
    model = load_model(os.path.join(os.path.abspath(
        os.path.dirname(__file__)), 'classification/model.h5'))

    breeds = []
    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'breeds.txt')) as f:
        lines = f.readlines()
        for line in lines:
            breeds.append(line.replace('\n', ''))

    image_bgr = cv2.imread(
        './data/raw/golden_retriever/golden_retriever_020.jpg')
    # image_bgr = cv2.imread('./sample/sample.jpg')

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(np.uint8(image_rgb))

    coordinates = detect_dogs.detect_image(image_pil)

    out = image_bgr
    for coordinate in coordinates:
        image_crop = image_bgr[coordinate[1]:coordinate[3], coordinate[0]:coordinate[2]]

        image_crop = cv2.resize(image_crop, (INPUT_SIZE, INPUT_SIZE)) / 255

        image_crop = np.reshape(image_crop, (1, INPUT_SIZE, INPUT_SIZE, 3))
        prediction = model.predict(image_crop)
        prediction_class = breeds[np.argmax(prediction)]

        # Draw label.
        y = coordinate[1] - 10 if coordinate[1] - \
            10 > 10 else coordinate[1] + 10
        cv2.putText(out, prediction_class, (coordinate[0], y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Draw renctangle.
        out = cv2.rectangle(out,
                            (coordinate[0], coordinate[1]),
                            (coordinate[2], coordinate[3]),
                            (0, 255, 0),
                            thickness=2)

    cv2.imwrite('./model_out.jpg', out)

    detect_dogs.close_session()
