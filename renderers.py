import cv2 as cv

import hsv_detector as hsv


def render_without_hsv(resized_image, results):
    """
    Renders the results without HSV classification.
    """
    for result in results:
        for box in result.boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Get the class name from the box
            class_id = int(box.cls[0])
            color_name = result.names[class_id]  # Get the class name
            confidence = box.conf[0]

            # UI Color logic
            colors = {"red": (0, 0, 255), "green": (0, 255, 0), "yellow": (0, 255, 255)}
            color_bgr = colors.get(color_name, (255, 255, 255))

            # Draw Bounding Box and Label
            cv.rectangle(resized_image, (x1, y1), (x2, y2), color_bgr, 3)
            label = f"{color_name} ({confidence:.2f})"
            cv.putText(
                resized_image,
                label,
                (x1, y1 - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                color_bgr,
                2,
            )


def render_with_hsv(resized_image, results):
    """
    Renders the results with HSV classification.
    """

    for result in results:
        for box in result.boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop for HSV classification
            roi = resized_image[y1:y2, x1:x2]

            # Check if ROI is valid (not empty)
            if roi.size == 0:
                continue

            light_class, pixel_count = hsv.classify_traffic_light(roi)

            if light_class == "none":
                continue

            # UI Color logic
            colors = {"red": (0, 0, 255), "green": (0, 255, 0), "yellow": (0, 255, 255)}
            color = colors.get(light_class, (255, 255, 255))

            # Draw Bounding Box and Label
            cv.rectangle(resized_image, (x1, y1), (x2, y2), color, 3)
            label = f"{light_class} ({box.conf[0]:.2f})"
            cv.putText(
                resized_image,
                label,
                (x1, y1 - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
