import colorsys
import glob
import os
import sys

import cv2
from keras import backend as K
from keras import preprocessing
from keras.layers import Input
from keras.models import load_model
from keras.utils import multi_gpu_model
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFont, ImageDraw

from model import DetectDogs

INPUT_SIZE = 299


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Invalid argments')
        print('> python3 yolo_crop.py [breeds.txt]')
        sys.exit()
    else:
        text_file = sys.argv[1]

    detect_dogs = DetectDogs()

    breeds = []
    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), text_file)) as f:
        lines = f.readlines()
        for line in lines:
            breeds.append(line.replace('\n', ''))

    data_dir = 'data/'
    raw_dir = data_dir + 'raw/'
    crop_dir = data_dir + 'crop/'

    for breed in breeds:
        print(breed)
        os.makedirs(crop_dir + breed, exist_ok=True)
        raw_images = glob.glob(raw_dir + breed + '/*.jpg')

        for raw_image in raw_images:
            image_bgr = cv2.imread(raw_image)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(np.uint8(image_rgb))

            coordinates = detect_dogs.detect_image(image_pil)

            count = 0
            # Windows
            save_path = crop_dir + breed + '/' + \
                os.path.splitext(raw_image)[0].split('\\')[-1] + '_' + str(count) + '.jpg'

            for coordinate in coordinates:
                image_crop = image_bgr[coordinate[1]:coordinate[3], coordinate[0]:coordinate[2]]
                cv2.imwrite(save_path, image_crop)
                count += 1

    detect_dogs.close_session()
