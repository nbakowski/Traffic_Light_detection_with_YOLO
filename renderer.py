from rich.progress import Progress
import cv2 as cv
from ultralytics.engine.model import Model

import hsv_detector
import settings


def render_start(
        capture: cv.VideoCapture,
        total_frame_count: int,
        out: cv.VideoWriter | None,
        uses_webcam: bool,
        name: str,
        model: Model
) -> None:
    progress = Progress() if not uses_webcam else None
    task = progress.add_task(f"Processing: {name}", total=total_frame_count) if progress else None

    if progress:
        progress.start()

    while True:
        is_running, frame = capture.read()
        if not is_running:
            break

        resized_image = cv.resize(frame, (settings.SCALED_IMAGE_WIDTH, settings.SCALED_IMAGE_HEIGHT))

        # YOLO Detection
        results = model.predict(source=resized_image,verbose=False)

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