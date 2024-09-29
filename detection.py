import os
import enum

from dataclasses import dataclass

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

torch.use_deterministic_algorithms(False)

from typing import Any

import numpy as np
import supervision as sv
from helpers import (combine_detections, load_grounding_dino, load_SAM)
from groundingdino.util.inference import Model
from segment_anything import SamPredictor

from ontology import CaptionOntology

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NmsSetting(str, enum.Enum):
    NONE = "no_nms"
    CLASS_SPECIFIC = "class_specific"
    CLASS_AGNOSTIC = "class_agnostic"

@dataclass
class Detection():
    ontology: CaptionOntology
    grounding_dino_model: Model
    sam_predictor: SamPredictor
    box_threshold: float
    text_threshold: float

    def __init__(
        self, ontology: CaptionOntology, box_threshold=0.35, text_threshold=0.25
    ):
        self.ontology = ontology
        self.grounding_dino_model = load_grounding_dino()
        self.sam_predictor = load_SAM()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def predict(self, input: Any) -> sv.Detections:
        # GroundingDINO predictions
        detections_list = []

        for _, description in enumerate(self.ontology.prompts()):
            # detect objects
            detections = self.grounding_dino_model.predict_with_classes(
                image=input,
                classes=[description],
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )

            detections_list.append(detections)

        detections = combine_detections(
            detections_list, overwrite_class_ids=range(len(detections_list))
        )

        # only keep the detection with smallest area
        # NOTE: this is just for finding label on the object, it will significantly reduce the SAM prediction time
        # if len(detections.xyxy) > 1:
        #     detections = smallest_detection(detections)
            # detections = most_confident_detection(detections)

        # SAM Predictions
        xyxy = detections.xyxy

        self.sam_predictor.set_image(input)
        result_masks = []
        for box in xyxy:
            masks, scores, _ = self.sam_predictor.predict(
                box=box, multimask_output=False
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])

        detections.mask = np.array(result_masks)

        # separate in supervision to combine detections and override class_ids
        return detections
