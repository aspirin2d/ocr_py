from detection import Detection
from ontology import CaptionOntology
from PIL import Image
import cv2
from cv2.typing import MatLike
import supervision as sv
import numpy as np
from cv2.barcode import BarcodeDetector
print("opencv version", cv2.__version__)

from paddleocr import PaddleOCR
from helpers import mask_to_polygon

from pyzbar.pyzbar import decode

ocr = PaddleOCR(use_angle_cls=True, lang="ch")
bd = BarcodeDetector('.cache/wechatcv/sr.prototxt',
                                 '.cache/wechatcv/sr.caffemodel')

def detect(image:MatLike, ontology:CaptionOntology) -> (sv.Detections |
None):
    detector = Detection(ontology=CaptionOntology({"label on a package":"label"}))
    detections = detector.predict(image)
    pass

def main(preview=True):
    # Create a detection object with the caption ontology
    # propmpt:class
    det = Detection(ontology=CaptionOntology({"label on a package":"label"}))
    image = cv2.imread("./images/input/test.jpg")
    detect(image, CaptionOntology({"label on a package":"label"}))

    # get the mask of detection
    detections = det.predict(image)
    if detections.mask is not None:
        for detection_idx in range(len(detections)):
            mask = detections.mask[detection_idx]

            # merge the masks into ONE polygon
            polygon =  mask_to_polygon(mask=mask)

            # if area of the polygon is too small(eg. 100x100), skip
            area = cv2.contourArea(polygon)
            if area < 10000:
                continue

            # cv2.polylines(image, [polygon], True, (0, 255, 0), 3)

            # approx_poly = cv2.approxPolyDP(polygon, 0.02*cv2.arcLength(polygon, True), True)
            # cv2.polylines(image, [approx_poly], True, (255, 0, 0), 3)

            # create obb
            rotated_rect = cv2.minAreaRect(polygon)
            box = cv2.boxPoints(rotated_rect)
            box = np.intp(box).astype(int)

            # cv2.polylines(image, [box], True, (0, 0, 255), 3)

            # crop the obb
            rotated = cv2.warpAffine(image, cv2.getRotationMatrix2D(
                rotated_rect[0], rotated_rect[2], 1.0), 
                (image.shape[1], image.shape[0]))

            cropped = cv2.getRectSubPix(rotated, 
                (int(rotated_rect[1][0]), int(rotated_rect[1][1])), 
                rotated_rect[0])

            cv2.imwrite(f"barcode_{str(detection_idx)}.jpg", cropped)
            decoded = decode(cv2.imread(f"barcode_{str(detection_idx)}.jpg", cv2.IMREAD_GRAYSCALE))
            print("decoded", decoded)

            # if width is larger than height, rotate the image 90 degree
            # if cropped.shape[0] > cropped.shape[1]:
            #     cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)

            # OCR the cropped image
            result = ocr.ocr(cropped, cls=True)[0]

            # draw the text boxes
            boxes = [line[0] for line in result]
            box_num = len(boxes)
            for i in range(box_num):
                box = np.reshape(np.array(boxes[i]), [-1, 1, 2]).astype(np.int64)
                cropped = cv2.polylines(np.array(cropped), [box], True, (0, 0,
                                                                         255), 2)
            cv2.imwrite(f"cropped_{str(detection_idx)}.jpg", cropped)

if __name__ == "__main__":
    main()
