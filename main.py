import logging
import sys

import cv2 as cv
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import file_operations as files
import renderer
import settings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def open_video_folder() -> None:
    files.open_folder("video")


def open_output_folder() -> None:
    files.open_folder("output")


class WorkerThread(QThread):
    """
    A generic worker thread to run heavy tasks (Webcam or File Scanning)
    without freezing the GUI.
    """

    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, target_function, *args):
        super().__init__()
        self.target_function = target_function
        self.args = args

    def run(self):
        try:
            self.target_function(*self.args)
        except Exception as e:
            self.error_signal.emit(str(e))
        finally:
            self.finished_signal.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.open_output_folder_button = QPushButton("Open Output Folder")
        self.open_video_folder_button = QPushButton("Open Source Folder")
        self.worker = None
        self.use_webcam_button = QPushButton("Use Webcam")
        self.scan_new_files_button = QPushButton("Scan New Files", self)
        self.scan_all_files_button = QPushButton("Scan All files", self)

        self.setWindowTitle("Traffic Light Detection with YOLO")
        self.setWindowIcon(QIcon("icon.png"))
        self.setFixedHeight(480)
        self.setFixedWidth(640)
        self.setStyleSheet("""
            QMainWindow {
                background-color: white;
                color: black;
            }
            QWidget {
                background-color: white;
                color: black;
            }
        """)

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Center picture
        center_picture = QLabel(self)
        center_picture.setPixmap(QPixmap("icon.png"))
        center_picture.setScaledContents(True)
        center_picture.setFixedSize(128, 128)
        center_picture.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Options label
        options_label = QLabel(self)
        options_label.setText(
            "This program allows for recognition of traffic "
            "lights using YOLO. You can choose a mode in which "
            "the program should run. Video files to analyze should be "
            "placed inside of the /video folder in the main application directory. "
            "The results can be found in the /output folder."
        )
        options_label.setStyleSheet("color: black; ")
        options_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        options_label.setWordWrap(True)
        options_label.setFixedSize(256, 256)

        # Buttons
        self.use_webcam_button.setFixedSize(150, 50)
        self.use_webcam_button.clicked.connect(self.start_webcam)
        self.use_webcam_button.setStyleSheet(
            "background-color: #ff0000; color: black; font-weight: bold;"
        )

        self.scan_new_files_button.setFixedSize(150, 50)
        self.scan_new_files_button.clicked.connect(self.start_scanning_new_files)
        self.scan_new_files_button.setStyleSheet(
            "background-color: #ffff00; color: black; font-weight: bold;"
        )

        self.scan_all_files_button.setFixedSize(150, 50)
        self.scan_all_files_button.clicked.connect(self.start_scanning_all_files)
        self.scan_all_files_button.setStyleSheet(
            "background-color: #00ff00; color: black; font-weight: bold;"
        )

        self.open_video_folder_button.setFixedSize(150, 50)
        self.open_video_folder_button.clicked.connect(open_video_folder)
        self.open_video_folder_button.setStyleSheet(
            "background-color: #cccccc; color: black;"
        )

        self.open_output_folder_button.setFixedSize(150, 50)
        self.open_output_folder_button.clicked.connect(open_output_folder)
        self.open_output_folder_button.setStyleSheet(
            "background-color: #cccccc; color: black;"
        )

        # Layout
        main_hbox = QHBoxLayout()

        # Folder buttons
        folder_buttons_hbox = QHBoxLayout()
        folder_buttons_hbox.addWidget(
            self.open_video_folder_button, alignment=Qt.AlignmentFlag.AlignCenter
        )
        folder_buttons_hbox.addWidget(
            self.open_output_folder_button, alignment=Qt.AlignmentFlag.AlignCenter
        )

        # Layout for labels and descriptions
        labels_vbox = QVBoxLayout()
        labels_vbox.addStretch()
        labels_vbox.addWidget(center_picture, alignment=Qt.AlignmentFlag.AlignCenter)
        labels_vbox.addWidget(options_label, alignment=Qt.AlignmentFlag.AlignCenter)
        labels_vbox.addLayout(folder_buttons_hbox)
        labels_vbox.addStretch()

        # Layout for buttons
        buttons_vbox = QVBoxLayout()
        buttons_vbox.addWidget(
            self.use_webcam_button, alignment=Qt.AlignmentFlag.AlignCenter
        )
        buttons_vbox.addWidget(
            self.scan_new_files_button, alignment=Qt.AlignmentFlag.AlignCenter
        )
        buttons_vbox.addWidget(
            self.scan_all_files_button, alignment=Qt.AlignmentFlag.AlignCenter
        )

        main_hbox.addLayout(labels_vbox)
        main_hbox.addLayout(buttons_vbox)

        central_widget.setLayout(main_hbox)

    def set_buttons_enabled(self, enabled: bool):
        """Disables buttons while a process is running."""
        self.use_webcam_button.setEnabled(enabled)
        self.scan_new_files_button.setEnabled(enabled)
        self.scan_all_files_button.setEnabled(enabled)

    def on_task_finished(self) -> None:
        """Called when a task is finished."""
        self.set_buttons_enabled(True)
        QMessageBox.information(
            self,
            "Success",
            "Operation was successful!\nCheck logs for more info.",
            QMessageBox.Ok,
        )

    def on_task_error(self, error_msg) -> None:
        """Called when a task fails."""
        self.set_buttons_enabled(True)
        QMessageBox.critical(self, "Error", f"An error occurred:\n{error_msg}")

    def start_webcam(self) -> None:
        capture = cv.VideoCapture(0)
        self.run_in_thread(
            renderer.render_start, capture, 0, None, True, "Webcam Feed", settings.MODEL
        )

    def start_scanning_new_files(self) -> None:
        self.run_in_thread(files.prep_files, False)

    def start_scanning_all_files(self) -> None:
        self.run_in_thread(files.prep_files, True)

    def run_in_thread(self, target, *args):
        """Helper to start the worker thread."""
        self.set_buttons_enabled(False)
        self.worker = WorkerThread(target, *args)
        self.worker.finished_signal.connect(self.on_task_finished)
        self.worker.error_signal.connect(self.on_task_error)
        self.worker.start()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
