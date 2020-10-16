import os
import cv2
import numpy as np


class DetectorDarknet:
    """ Adaptor for the original Yolo implementation (AlexeyAB/darknet) """

    def __init__(self, config, fn_tracking_convert=None):
        # Env var used in python wrapper to load .so library
        os.environ["DARKNET_PATH"] = config["darknet_path"]
        from integrations.yolo import darknet  # noqa

        # Before calling this script, must also add path with libcudart.so
        # e.g: export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64

        self.darknet = darknet
        self.detection_threshold = config["detection_threshold"]
        self.nms_threshold = config["nms_threshold"]

        self.network, self.class_names, class_colors = darknet.load_network(
            config["config_file"], config["data_file"], config["weights_file"], batch_size=1
        )

        # Darknet doesn't accept numpy images.
        # Create one with image we reuse for each detect
        self.width = darknet.network_width(self.network)
        self.height = darknet.network_height(self.network)
        self.darknet_image = darknet.make_image(self.width, self.height, 3)

    def detect(self, frame, rescale_detections=True, recolor=False, min_size=16, blacklist=None):
        orig_height, orig_width = frame.shape[:2]
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.darknet.copy_image_from_bytes(self.darknet_image, frame.tobytes())
        detections = self.darknet.detect_image_combine_classes(
            self.network,
            self.class_names,
            self.darknet_image,
            thresh=self.detection_threshold,
            nms=self.nms_threshold,
            combine=[2, 7],  # Combine truck into car (see indices in coco.names)
        )
        w_scale = orig_width / self.width
        h_scale = orig_height / self.height
        dets = []
        for d in detections:
            if blacklist is not None and d[0] in blacklist:
                continue
            xc, yc, w, h = d[2]
            if rescale_detections:
                xc *= w_scale
                yc *= h_scale
                w *= w_scale
                h *= h_scale
            if w < min_size:
                continue
            dets.append({"detection": (xc, yc, w, h), "label": d[0], "p": d[1]})
        if recolor:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return dets, frame
