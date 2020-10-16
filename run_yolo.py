# %%
import sys
import yaml
import time

from norfair.video import Video
from norfair.tracker import Tracker
from norfair.drawing import draw_tracked_objects, draw_points, draw_debug_metrics, Color

from integrations.yolo.detector_darknet import DetectorDarknet

# from integrations.yolo.detector_pytorch import DetectorYoloPytorch

# Required python tensorrt, usually compiled for python 3.6 at system level
# from integrations.yolo.detector_trt import DetectorYoloTRT

from integrations.yolo.yolo_adaptor import YoloAdaptor


def draw_margins(frame, config):
    zones = [
        frame[: config["top_margin"], :],
        frame[config["top_margin"] : -config["bottom_margin"], : config["left_margin"]],
    ]
    # Indexings below don't work with margin = 0
    if config["bottom_margin"] > 0:
        zones.append(frame[-config["bottom_margin"] :, :])
    if config["right_margin"] > 0:
        zones.append(
            frame[config["top_margin"] : -config["bottom_margin"], -config["right_margin"] :]
        )
    # Darken dividing each pixel by 2
    for zone in zones:
        zone[:, :, :] >>= 1


def validate_box_position(box, frame, config):
    """ Discard boxes if some part is out of the scene """
    return not (
        box[0][0] < config["left_margin"]  # x1
        or box[0][1] < config["top_margin"]  # y1
        or box[1][0] >= frame.shape[1] - config["right_margin"]  # x2
        or box[1][1] >= frame.shape[0] - config["bottom_margin"]  # y2
    )


# %%
with open("config.yml", "r") as stream:
    # Not using Loader=yaml.FullLoader since it doesn't work on jetson PyYAML version
    config = yaml.load(stream)

# Yolo implementation to use
detector = DetectorDarknet({**config["yolo_darknet"], **config["yolo_generic"]})

# detector = DetectorYoloPytorch({**config["yolo_pytorch"], **config["yolo_generic"]})

# detector = DetectorYoloTRT({**config["yolo_trt"], **config["yolo_generic"]})

pose_adaptor = YoloAdaptor(config["yolo_generic"])

# In Norfair we trust
tracker = Tracker(
    distance_function=pose_adaptor.keypoints_distance,
    detection_threshold=pose_adaptor.detection_threshold,
    distance_threshold=pose_adaptor.distance_threshold,
    hit_inertia_min=10,
    hit_inertia_max=80,
    point_transience=10,
)

# Video handler (Norfair)
video = Video(
    input_path=sys.argv[1],
    output_path=config["general"]["output_folder"],  # , codec_fourcc="avc1")
)

timer_yolo = 0.0  # Reset to 0.0 after first frame to avoid counting model loading
timer_tracker = 0.0
timer_drawing = 0.0
for k, frame in enumerate(video):

    tick = time.time()

    # Filter parts to track, and also keep detected pose tracked_scores for later use
    detections_yolo, frame_preprocessed = detector.detect(
        frame, rescale_detections=True, blacklist=[], min_size=config["yolo_generic"]["min_size"],
    )
    for d in detections_yolo:
        pose_adaptor.add_detection_bbox(d)
    detections_inframe = [
        d for d in detections_yolo if validate_box_position(d["bbox"], frame, config["general"])
    ]
    timer_yolo += time.time() - tick

    # Tracker update
    tick = time.time()
    detections_tracker = [pose_adaptor.yolo_to_tracking(d) for d in detections_inframe]
    tracked_objects = tracker.update(
        detections_tracker, period=config["general"]["inference_period"]
    )
    timer_tracker += time.time() - tick

    # Drawing functions
    tick = time.time()
    draw_margins(frame, config["general"])
    if config["debug"]["draw_detections"]:  # Using yolo detections
        # draw_points(frame, detections_tracker)
        pose_adaptor.draw_raw_detections(frame, detections_inframe)
    if config["debug"]["draw_predictions"]:
        draw_tracked_objects(frame, tracked_objects, id_size=0)
    if config["debug"]["draw_tracking_ids"]:
        draw_tracked_objects(frame, tracked_objects, draw_points=False, id_thickness=1)
    if config["debug"]["draw_tracking_debug"]:
        draw_debug_metrics(frame, tracked_objects)

    video.write(frame)
    timer_drawing += time.time() - tick

    # Reset counters after first frame to avoid counting model loading
    if k == 0:
        timer_yolo = 0.0
        timer_tracker = 0.0
        timer_drawing = 0.0

if config["debug"]["profiler"]:
    # No need to divide between (k+1) - counters reset on k==0
    timer_total = timer_yolo + timer_tracker + timer_drawing
    print(f"Avg total time/frame:\t{timer_total / k:.4f}s\t| FPS: {k / timer_total:.1f}")
    print(f"Avg yolo time/frame:\t{timer_yolo / k:.4f}s\t| FPS: {k / timer_yolo:.1f}")
    print(f"Avg tracker time/frame:\t{timer_tracker / k:.4f}s\t| FPS: {k / timer_tracker:.1f}")
    print(f"Avg drawing time/frame:\t{timer_drawing / k:.4f}s\t| FPS: {k / timer_drawing:.1f}")
