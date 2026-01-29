import cv2 as cv
from rich.progress import Progress
from ultralytics.engine.model import Model

import renderers as r
import settings
from settings import RENDERMODE


def render_start(
    capture: cv.VideoCapture,
    total_frame_count: int,
    out: cv.VideoWriter | None,
    uses_webcam: bool,
    name: str,
    model: Model,
) -> None:
    """
    Contains the main processing loop.
    Based on the selected render mode, it will call the appropriate renderer function.
    """
    progress = Progress() if not uses_webcam else None
    task = (
        progress.add_task(f"Processing: {name}", total=total_frame_count)
        if progress
        else None
    )

    if progress:
        progress.start()

    while True:
        is_running, frame = capture.read()
        if not is_running:
            break

        resized_image = cv.resize(
            frame, (settings.SCALED_IMAGE_WIDTH, settings.SCALED_IMAGE_HEIGHT)
        )

        # YOLO Detection
        match RENDERMODE:
            case settings.RenderMode.WITHOUT_HSV:
                results = model.predict(source=resized_image, verbose=False, conf=0.1)
                r.render_without_hsv(resized_image, results)
            case settings.RenderMode.WITH_HSV:
                results = model.predict(
                    source=resized_image, verbose=False, classes=[9], conf=0.25
                )
                r.render_with_hsv(resized_image, results)


        if progress and task is not None:
            progress.update(task, advance=1)

        if uses_webcam:
            cv.putText(
                resized_image,
                "Press ESC to exit.",
                (25, 50),
                cv.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                2,
            )
            cv.imshow(name, resized_image)
            if cv.waitKey(1) & 0xFF == 27:
                break
        else:
            if out is not None:
                out.write(resized_image)

    # Cleanup
    if progress:
        progress.stop()
    if not uses_webcam and out is not None:
        out.release()
    capture.release()
    cv.destroyAllWindows()
