import os
import urllib.request
import cv2

import numpy as np
import supervision as sv
import torch
from groundingdino.util.inference import Model
from segment_anything import SamPredictor, sam_model_registry

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not torch.cuda.is_available():
    print("WARNING: CUDA not available. GroundingDINO will run very slowly.")

def crop_obb(img, rect):
    # get the parameter of the small rectangle
    center = rect[0]
    size = rect[1]
    angle = rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]
    # print("width: {}, height: {}".format(width, height))

    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (width, height))

    img_crop = cv2.getRectSubPix(img_rot, size, center)
    return img_crop

# find the most confident detection in detections
def most_confident_detection(detections):
    if detections.confidence is None:
        return None

    max_idx = np.argmax(detections.confidence)
    return sv.Detections(
        xyxy=detections.xyxy[max_idx : max_idx + 1],
        mask=detections.mask[max_idx : max_idx + 1] if detections.mask is not None else None,
        confidence=detections.confidence[max_idx : max_idx + 1],
        class_id=detections.class_id[max_idx : max_idx + 1] if detections.class_id is not None else None,
        tracker_id=detections.tracker_id[max_idx : max_idx + 1] if detections.tracker_id is not None else None,
    )

# find the detection with smallest area in detections
def smallest_detection(detections):
    if detections.xyxy is None:
        return None

    areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * (detections.xyxy[:, 3] - detections.xyxy[:, 1])
    min_idx = np.argmin(areas)
    return sv.Detections(
        xyxy=detections.xyxy[min_idx : min_idx + 1],
        mask=detections.mask[min_idx : min_idx + 1] if detections.mask is not None else None,
        confidence=detections.confidence[min_idx : min_idx + 1],
        class_id=detections.class_id[min_idx : min_idx + 1] if detections.class_id is not None else None,
        tracker_id=detections.tracker_id[min_idx : min_idx + 1] if detections.tracker_id is not None else None,
    )

def combine_detections(detections_list, overwrite_class_ids):
    if len(detections_list) == 0:
        return sv.Detections.empty()

    if overwrite_class_ids is not None and len(overwrite_class_ids) != len(
        detections_list
    ):
        raise ValueError(
            "Length of overwrite_class_ids must match the length of detections_list."
        )

    xyxy = []
    mask = []
    confidence = []
    class_id = []
    tracker_id = []

    for idx, detection in enumerate(detections_list):
        xyxy.append(detection.xyxy)

        if detection.mask is not None:
            mask.append(detection.mask)

        if detection.confidence is not None:
            confidence.append(detection.confidence)

        if detection.class_id is not None:
            if overwrite_class_ids is not None:
                # Overwrite the class IDs for the current Detections object
                class_id.append(
                    np.full_like(
                        detection.class_id, overwrite_class_ids[idx], dtype=np.int64
                    )
                )
            else:
                class_id.append(detection.class_id)

        if detection.tracker_id is not None:
            tracker_id.append(detection.tracker_id)

    xyxy = np.vstack(xyxy)
    mask = np.vstack(mask) if mask else None
    confidence = np.hstack(confidence) if confidence else None
    class_id = np.hstack(class_id) if class_id else None
    tracker_id = np.hstack(tracker_id) if tracker_id else None

    return sv.Detections(
        xyxy=xyxy,
        mask=mask,
        confidence=confidence,
        class_id=class_id,
        tracker_id=tracker_id,
    )


def load_grounding_dino():
    CACHE_DIR = os.path.expanduser("./.cache")

    GROUDNING_DINO_CACHE_DIR = os.path.join(CACHE_DIR, "groundingdino")

    GROUNDING_DINO_CONFIG_PATH = os.path.join(
        GROUDNING_DINO_CACHE_DIR, "cfg_odvg.py"
        # GROUDNING_DINO_CACHE_DIR, "GroundingDINO_SwinB_cfg.py"
    )
    GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(
        GROUDNING_DINO_CACHE_DIR, "gdinot-1.8m-odvg.pth"
        # GROUDNING_DINO_CACHE_DIR, "groundingdino_swinb_cogcoor.pth"
    )

    try:
        print("trying to load grounding dino directly")
        grounding_dino_model = Model(
            model_config_path=GROUNDING_DINO_CONFIG_PATH,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
            device=str(DEVICE),
        )
        return grounding_dino_model
    except Exception:
        print("downloading DINO model weights")
        if not os.path.exists(GROUDNING_DINO_CACHE_DIR):
            os.makedirs(GROUDNING_DINO_CACHE_DIR)

        if not os.path.exists(GROUNDING_DINO_CHECKPOINT_PATH):
            # url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"
            url = "https://github.com/longzw1997/Open-GroundingDino/releases/download/v0.1.0/gdinot-1.8m-odvg.pth"
            urllib.request.urlretrieve(url, GROUNDING_DINO_CHECKPOINT_PATH)

        if not os.path.exists(GROUNDING_DINO_CONFIG_PATH):
            url = "https://raw.githubusercontent.com/longzw1997/Open-GroundingDino/refs/heads/main/config/cfg_odvg.py"
            # url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/refs/heads/main/groundingdino/config/GroundingDINO_SwinB_cfg.py"
            urllib.request.urlretrieve(url, GROUNDING_DINO_CONFIG_PATH)

        grounding_dino_model = Model(
            model_config_path=GROUNDING_DINO_CONFIG_PATH,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
            device=str(DEVICE),
        )

        # grounding_dino_model.to(DEVICE)
        return grounding_dino_model


def load_SAM():
    # Check if segment-anything library is already installed
    CACHE_DIR = os.path.expanduser("./.cache")
    SAM_CACHE_DIR = os.path.join(CACHE_DIR, "segment_anything")
    SAM_CHECKPOINT_PATH = os.path.join(SAM_CACHE_DIR, "sam_vit_h_4b8939.pth")

    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

    # Create the destination directory if it doesn't exist
    os.makedirs(os.path.dirname(SAM_CHECKPOINT_PATH), exist_ok=True)

    # Download the file if it doesn't exist
    if not os.path.isfile(SAM_CHECKPOINT_PATH):
        print("downloading SAM model weights")
        urllib.request.urlretrieve(url, SAM_CHECKPOINT_PATH)

    SAM_ENCODER_VERSION = "vit_h"

    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(
        device=DEVICE
    )
    sam_predictor = SamPredictor(sam)

    return sam_predictor
