import logging
import os

import cv2 as cv

import settings
from renderer import render_start


def scan_for_existing_files() -> list[str]:
    """
    Scans for existing files in the output directory.
    """
    new_files = []
    for name in os.scandir("video"):
        if os.path.exists(f"output/{name.name}_processed.mp4"):
            logging.info(f"File {name.name}_processed.mp4 already exists!")
        else:
            new_files.append(name.name)
    logging.info(f"Found {len(new_files)} new file(s)!")
    return new_files


def manage_directories() -> None:
    """
    Manages the directories for the application.
    If the directories do not exist, they are created.
    """
    directories_list = ["video/", "output/"]
    for directory in directories_list:
        if not os.path.isdir(directory):
            os.makedirs(directory)
            logging.info(f"Created the {directory} directory.")


def prep_files(scan_all: bool) -> None:
    """
    Prepares the files for processing.
    If scan_all is True, all files are processed.
    If scan_all is False, only new files are processed.
    """
    manage_directories()

    files_to_scan = []
    if not scan_all:
        files_to_scan = scan_for_existing_files()

    for name in os.scandir("video"):
        if name.name not in files_to_scan:
            continue

        capture = cv.VideoCapture(name.path)
        total_frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

        if not capture.isOpened():
            raise Exception

        out_path = f"output/{name.name}_processed.mp4"

        if os.path.exists(out_path):
            try:
                os.remove(out_path)
                logging.info(f"Removed existing output: {out_path}")
            except Exception as e:
                logging.error(f"Could not remove existing output {out_path}: {e}")
                continue

        out = cv.VideoWriter(
            out_path,
            settings.CODEC,
            30,
            (settings.SCALED_IMAGE_WIDTH, settings.SCALED_IMAGE_HEIGHT),
        )

        if not out.isOpened():
            logging.error(f"Could not open video writer for {out_path}")
            continue

        render_start(capture, total_frame_count, out, False, name.path, settings.MODEL)
