import cv2
import numpy as np

from norfair.tracker import Detection
from norfair.drawing import Color


class YoloAdaptor:
    def __init__(self, config):
        self.detection_threshold = config["detection_threshold"]
        self.distance_threshold = config["distance_threshold"]

    # Select tracking points from yolo detections (overrides DetectorDarknet._yolo_to_bbox())
    def yolo_to_tracking(self, detection_yolo):
        x, y, w, h = detection_yolo["detection"]  # Yolo output: x_center, y_center, width, height
        x_left, x_right = x - w / 2, x + w / 2
        y_top = y - h / 2
        # Using centroid + center-top, and both top corners:
        # Rationale: top points are more stable (e.g: wheels usually get occluded),
        # and central points are more robust on lateral occlusions (e.g: passing behind a truck)
        # This jumpy behavior of lateral and bottom points is poison for the kalman filter
        return Detection(
            np.array(((x, y), (x, y_top), (int(x_left), int(y_top)), (int(x_right), int(y_top)))),
            data={"label": detection_yolo["label"], "p": detection_yolo["p"]},
        )

    def add_detection_bbox(self, detection_yolo):
        x, y, w, h = detection_yolo["detection"]  # Yolo output: x_center, y_center, width, height
        x_left, x_right = x - w / 2, x + w / 2
        y_top, y_bottom = y - h / 2, y + h / 2
        detection_yolo["bbox"] = ((int(x_left), int(y_top)), (int(x_right), int(y_bottom)))

    def keypoints_distance(self, detection, tracked_object):
        ref_size_x = np.max(detection.points[:, 0]) - np.min(detection.points[:, 0])
        ref_size_y = np.max(detection.points[:, 1]) - np.min(detection.points[:, 1])
        ref_size = max(ref_size_x, ref_size_y)
        return np.linalg.norm(detection.points - tracked_object.estimate) / ref_size

    def draw_raw_detections(self, frame, detections_yolo):
        for d in detections_yolo:
            if "bbox" not in d:
                self.add_detection_bbox(d)
            p1, p2 = d["bbox"]
            bbox = (tuple(p1), tuple(p2))
            label = d["label"]
            p = float(d["p"])
            color = Color.white
            cv2.rectangle(frame, bbox[0], bbox[1], color, 1)
            cv2.putText(
                frame,
                f"{label}: {p:.0f}",
                (bbox[0][0], bbox[0][1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
