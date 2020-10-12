import cv2
import numpy as np

from norfair.drawing import Color


class YoloAdaptor:
    def __init__(self, config):
        self.detection_threshold = config["detection_threshold"]
        self.distance_threshold = config["distance_threshold"]

    def keypoints_distance(self, detected_pose, tracked_pose):
        detected_points = detected_pose.points
        estimated_pose = tracked_pose.estimate
        min_box_size = min(
            max(
                detected_points[1][0] - detected_points[0][0],  # x2 - x1
                detected_points[1][1] - detected_points[0][1],  # y2 - y1
                1,
            ),
            max(
                estimated_pose[1][0] - estimated_pose[0][0],  # x2 - x1
                estimated_pose[1][1] - estimated_pose[0][1],  # y2 - y1
                1,
            ),
        )
        mean_distance_normalized = (
            np.mean(np.linalg.norm(detected_points - estimated_pose, axis=1)) / min_box_size
        )
        return mean_distance_normalized

    def draw_raw_detections(self, frame, detections):
        for d in detections:
            p1, p2 = d.points.astype(int)
            bbox = (tuple(p1), tuple(p2))
            label = d.data["label"]
            p = float(d.data["p"])
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
