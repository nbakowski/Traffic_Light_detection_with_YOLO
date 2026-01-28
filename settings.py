import os
import cv2
from ultralytics import YOLO

MODEL = YOLO("LISA.pt")
SCALED_IMAGE_WIDTH, SCALED_IMAGE_HEIGHT = 1270, 720
if os.name == "nt":
    CODEC = cv2.VideoWriter.fourcc(*"mp4v")
else:
    CODEC = cv2.VideoWriter.fourcc(*"avc1")