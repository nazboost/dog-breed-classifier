"""
Save images by cropping dogs detected with yolo for CNN.

Don't use this file (because file organize rewuired).
"""

import colorsys
import os

import cv2
from keras import backend as K
from keras.layers import Input
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image, ImageFont, ImageDraw

from yolo.yolo_model import yolo_eval, yolo_body, tiny_yolo_body
from yolo.yolo_utils import letterbox_image

from model import DetectDogs

if __name__ == "__main__":
    detect_dogs = DetectDogs()

    data_dir = './data/'
    raw_dir = data_dir + 'raw/'
    crop_dir = data_dir + 'crop/'

    breeds = []
    breeds_data = open(os.path.join(os.path.abspath(
        os.path.dirname(__file__)), 'breeds.txt'), 'r')
    lines = breeds_data.readlines()
    for line in lines:
        breeds.append(line.replace('\n', ''))

    for breed in breeds:
        print(breed)
        os.makedirs(crop_dir + breed, exist_ok=True)

        images = os.listdir(raw_dir + breed + '/')
        for image in images:
            image_pil = Image.open(raw_dir + breed + '/' + image)
            coordinates = detect_dogs.detect_image(image_pil)

            image_rgb = img_to_array(image_pil)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            for i, coordinate in enumerate(coordinates):
                save_path = crop_dir + breed + '/' + \
                    os.path.splitext(image)[0] + '_' + str(i) + '.jpg'

                cv2.imwrite(save_path,
                            image_bgr[coordinate[1]:coordinate[3],
                                      coordinate[0]:coordinate[2],
                                      :])
