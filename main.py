import os
import cv2 as cv
import logging
from rich.progress import Progress
from ultralytics import YOLO
import hsv_detector

# Configuration
MODEL = YOLO("yolo26s.pt")
SCALED_IMAGE_WIDTH, SCALED_IMAGE_HEIGHT = 1270, 720

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def exit_program() -> bool:
    os.system("clear")
    print("See you next time!")
    input("Press the Enter key to continue...")
    os.system("clear")
    return True


def manage_directories() -> None:
    directories_list = ["video/", "output/"]
    for directory in directories_list:
        if not os.path.isdir(directory):
            os.makedirs(directory)
            logging.info(f"Created the {directory} directory.")
            input("Press the Enter key to continue...")


def scan_for_existing_files() -> list[str]:
    new_files = []
    for name in os.scandir("video"):
        if os.path.exists(f"output/{name.name}_processed.mp4"):
            logging.info(f"File {name.name}_processed.mp4 already exists!")
        else:
            new_files.append(name.name)
    logging.info(f"Found {len(new_files)} new file(s)!")
    input("Press the Enter key to continue...")
    return new_files


def prep_files(scan_all: bool) -> None:
    if not scan_all:
        files_to_scan = scan_for_existing_files()

    for name in os.scandir("video"):
        if not scan_all:
            if name.name not in files_to_scan:
                continue

        capture = cv.VideoCapture(name.path)
        total_frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

        if not capture.isOpened():
            raise Exception

        # build output path once
        out_path = f"output/{name.name}_processed.mp4"

        # If file exists and we intend to (re)create it, remove it first so AVFoundation/OpenCV can write.
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
                logging.info(f"Removed existing output: {out_path}")
            except Exception as e:
                logging.error(f"Could not remove existing output {out_path}: {e}")
                # if the file couldn't be removed, skip this file to avoid AVAssetWriter errors
                continue

        fourcc = cv.VideoWriter.fourcc(*"avc1")
        out = cv.VideoWriter(
            out_path,
            fourcc,
            30,
            (SCALED_IMAGE_WIDTH, SCALED_IMAGE_HEIGHT),
        )

        if not out.isOpened():
            logging.error(f"Could not open video writer for {out_path}")
            continue

        render_start(capture, total_frame_count, out, False, name.path)


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


def render_start(
        capture: cv.VideoCapture,
        total_frame_count: int,
        out: cv.VideoWriter,
        uses_webcam: bool,
        name: str,
) -> None:
    progress = Progress() if not uses_webcam else None
    task = progress.add_task(f"Processing: {name}", total=total_frame_count) if progress else None

    if progress:
        progress.start()

    while True:
        is_running, frame = capture.read()
        if not is_running:
            break

        resized_image = cv.resize(frame, (SCALED_IMAGE_WIDTH, SCALED_IMAGE_HEIGHT))

        # YOLO Detection
        results = MODEL.predict(source=resized_image, verbose=False, device="cpu", classes=[9])

        for result in results:
            for box in result.boxes:
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Crop for HSV classification
                light_roi = resized_image[y1:y2, x1:x2]

                # Check if ROI is valid (not empty)
                if light_roi.size == 0:
                    continue

                light_class, pixel_count = hsv_detector.classify_traffic_light(light_roi)

                if light_class == "none":
                    continue

                # UI Color logic
                colors = {"red": (0, 0, 255), "green": (0, 255, 0), "yellow": (0, 255, 255)}
                color = colors.get(light_class, (255, 255, 255))

                # Draw Bounding Box and Label
                cv.rectangle(resized_image, (x1, y1), (x2, y2), color, 3)
                label = f"{light_class} ({box.conf[0]:.2f})"
                cv.putText(resized_image, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if progress and task is not None:
            progress.update(task, advance=1)

        if uses_webcam:
            cv.putText(resized_image, "Press ESC to exit.", (25, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            cv.imshow(name, resized_image)
            if cv.waitKey(1) & 0xFF == 27:
                break
        else:
            out.write(resized_image)

    # Cleanup
    if progress:
        progress.stop()
    if not uses_webcam:
        out.release()
    capture.release()
    cv.destroyAllWindows()


def main():
    while True:
        # os.system("clear")
        print("--- YOLO Traffic Light System ---")
        print("1 - Webcam.")
        print("2 - Files.")
        print("3 - Exit.")

        instruction = get_value([1, 2, 3])

        if instruction == 1:
            capture = cv.VideoCapture(0)
            render_start(capture, 0, None, True, "Webcam Feed")
        elif instruction == 2:
            manage_directories()
            while True:
                # os.system("clear")
                print("1 - Scan only new files.")
                print("2 - Scan all files.")
                print("3 - Show video folder.")
                print("4 - Show output folder.")
                print("5 - Back.")
                print("6 - Exit.")

                choice = get_value([1, 2, 3, 4, 5, 6])
                if choice == 1:
                    prep_files(False)
                elif choice == 2:
                    prep_files(True)
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