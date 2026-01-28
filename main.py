import logging
import os

import cv2 as cv

import file_operations as files
import renderer
import settings
import click

# Configuration
os.environ["USE_NNPACK"] = "0"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def exit_program() -> bool:
    click.clear()
    print("See you next time!")
    input("Press the Enter key to continue...")
    click.clear()
    return True


def show_files(directory_path: str, folder_name: str) -> None:
    print(f"{folder_name} folder:")
    for name in os.listdir(directory_path):
        print(name)
    input("Press the Enter key to continue...")


def get_value(possible_inputs: list[int]) -> int:
    while True:
        try:
            value = int(input("Choice: "))
            if value not in possible_inputs:
                raise ValueError
            return value
        except ValueError:
            print("Invalid input! Please choose from", possible_inputs)


def main():
    while True:
        # click.clear()
        print("--- YOLO Traffic Light System ---")
        print("1 - Webcam.")
        print("2 - Files.")
        print("3 - Exit.")

        instruction = get_value([1, 2, 3])

        if instruction == 1:
            capture = cv.VideoCapture(0)
            renderer.render_start(capture, 0, None, True, "Webcam Feed", settings.MODEL)
        elif instruction == 2:
            files.manage_directories()
            while True:
                click.clear()
                print("1 - Scan only new files.")
                print("2 - Scan all files.")
                print("3 - Show video folder.")
                print("4 - Show output folder.")
                print("5 - Back.")
                print("6 - Exit.")

                choice = get_value([1, 2, 3, 4, 5, 6])
                if choice == 1:
                    files.prep_files(False)
                elif choice == 2:
                    files.prep_files(True)
                elif choice == 3:
                    show_files("video", "Video")
                elif choice == 4:
                    show_files("output", "Output")
                elif choice == 5:
                    break
                else:
                    if exit_program(): return
        else:
            if exit_program(): return


if __name__ == "__main__":
    main()