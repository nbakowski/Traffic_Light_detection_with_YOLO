import os
from enum import Enum

import cv2
from ultralytics.models import YOLO


class RenderMode(Enum):
    WITH_HSV = ("with_hsv",)
    WITHOUT_HSV = ("without_hsv",)


MODEL = YOLO("models/COCO128_YOLO26_COLLAB.pt")
RENDERMODE = RenderMode.WITH_HSV
SCALED_IMAGE_WIDTH, SCALED_IMAGE_HEIGHT = 1270, 720
if os.name == "nt":
    CODEC = cv2.VideoWriter.fourcc(*"mp4v")
else:
    CODEC = cv2.VideoWriter.fourcc(*"avc1")
