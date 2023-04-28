import cv2
import numpy as np
from PIL import ImageFilter, ImageOps
import random


class GaussianBlur(object):
    def __init__(self, radius_min=0.1, radius_max=2.0):
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img


class Solarization(object):
    def __call__(self, img):
        return ImageOps.solarize(img)
