from typing import Any

import cv2 as cv
import numpy as np


def detect_red_light(hsv: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    Detects red light in the given HSV image.
    """
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv.inRange(hsv, lower_red2, upper_red2)
    return cv.bitwise_or(mask1, mask2)


def detect_yellow_light(hsv: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    Detects yellow light in the given HSV image.
    """
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    return cv.inRange(hsv, lower_yellow, upper_yellow)


def detect_green_light(hsv: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    Detects green light in the given HSV image.
    """
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([90, 255, 255])
    return cv.inRange(hsv, lower_green, upper_green)


def classify_traffic_light(roi: np.ndarray[Any, Any]) -> tuple[str, int]:
    """
    Classifies the traffic light in the given ROI.
    """
    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

    red_mask = detect_red_light(hsv)
    yellow_mask = detect_yellow_light(hsv)
    green_mask = detect_green_light(hsv)

    red_pixels = cv.countNonZero(red_mask)
    yellow_pixels = cv.countNonZero(yellow_mask)
    green_pixels = cv.countNonZero(green_mask)

    max_pixels = max(red_pixels, yellow_pixels, green_pixels)

    if max_pixels < 25:
        return "none", 0

    if red_pixels == max_pixels:
        return "red", red_pixels
    elif yellow_pixels == max_pixels:
        return "yellow", yellow_pixels
    else:
        return "green", green_pixels
