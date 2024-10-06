import argparse
import os

import cv2
import numpy as np
from paddleocr import PaddleOCR

from detection import Detection
from helpers import image_resize, mask_to_polygon, overlay
from ontology import CaptionOntology

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def detect(args):
    ontology = CaptionOntology({args.prt:args.cls})
    detection = Detection(ontology, args.bth, args.tth)

    image = cv2.imread(args.img)
    result = detection.predict(image)

    if args.flt > 0:
    # if detection is in another detection, remove one 
        for i in range(len(result.xyxy)):
            for j in range(len(result.xyxy)):
                if i != j:
                    if (result.xyxy[i][0] >= result.xyxy[j][0] 
                        and result.xyxy[i][1] >= result.xyxy[j][1] 
                        and result.xyxy[i][2] <= result.xyxy[j][2] 
                        and result.xyxy[i][3] <= result.xyxy[j][3]):
                        if args.flt == 1: # keep outter box
                            result.xyxy[i] = [0, 0, 0, 0]
                        else: # keep inner box
                            result.xyxy[j] = [0, 0, 0, 0]
    
    for i in range(len(result.xyxy)):
        # if area of the polygon is too small(eg. 100x100), skip
        area = (result.xyxy[i][2] - result.xyxy[i][0]) * (result.xyxy[i][3] - result.xyxy[i][1])
        if area < 10000:
            continue

        box = result.xyxy[i]
        mask = result.mask[i]

        if args.dbb:
            # draw bounding box, default is false
            box = [int(i) for i in box]
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 200, 0), 2)

        if args.dmk:
            # draw segments mask, default is false
            image = overlay(image, mask, (200, 0, 0), 0.3)
    
        if args.dbd:
            # draw segments border
            polygon =  mask_to_polygon(mask=mask)
            cv2.polylines(image, [polygon], True, (0, 200, 0), 3)
        
        if args.obb:
            polygon =  mask_to_polygon(mask=mask)
            rotated_rect = cv2.minAreaRect(polygon)
            box = cv2.boxPoints(rotated_rect)
            box = np.intp(box).astype(int)
            cv2.polylines(image, [box], True, (0, 0, 200), 3)

    if args.prv is True:
        # preview the detection result, default is true
        cv2.imshow('result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif args.prv is not None:
        # if args.prv is a path, save the detection result to the path
        cv2.imwrite(args.prv, image)

def crop(args):
    ontology = CaptionOntology({args.prt:args.cls})
    detection = Detection(ontology, args.bth, args.tth)

    image = cv2.imread(args.img)
    result = detection.predict(image)

    if args.flt > 0:
    # if detection is in another detection, remove one 
        for i in range(len(result.xyxy)):
            for j in range(len(result.xyxy)):
                if i != j:
                    if (result.xyxy[i][0] >= result.xyxy[j][0] 
                        and result.xyxy[i][1] >= result.xyxy[j][1] 
                        and result.xyxy[i][2] <= result.xyxy[j][2] 
                        and result.xyxy[i][3] <= result.xyxy[j][3]):
                        if args.flt == 1: # keep outter box
                            result.xyxy[i] = [0, 0, 0, 0]
                        else: # keep inner box
                            result.xyxy[j] = [0, 0, 0, 0]
    

    cropped_list = []
    for i in range(len(result.xyxy)):
        # if area of the polygon is too small(eg. 100x100), skip
        area = (result.xyxy[i][2] - result.xyxy[i][0]) * (result.xyxy[i][3] - result.xyxy[i][1])
        if area < 10000:
            continue

        box = result.xyxy[i]
        mask = result.mask[i]

        if args.rot is False:
            # crop with bounding box
            box = [int(i) for i in box]
            cropped = image[box[1]:box[3], box[0]:box[2]]
            # resize to 640x640
            resized = image_resize(cropped)
            cropped_list.append(resized)
        else:
            # crop with oriented bounding box
            polygon =  mask_to_polygon(mask=mask)
            rotated_rect = cv2.minAreaRect(polygon)
            box = cv2.boxPoints(rotated_rect)
            box = np.intp(box).astype(int)

            # rotate the origin image
            rotated = cv2.warpAffine(image, cv2.getRotationMatrix2D(
                rotated_rect[0], rotated_rect[2], 1.0), 
                (image.shape[1], image.shape[0]))

            # crop the obb
            cropped = cv2.getRectSubPix(rotated, 
                (int(rotated_rect[1][0]), int(rotated_rect[1][1])), 
                rotated_rect[0])
            # if cropped width is larger than height, rotate the image 90 degree
            # if cropped.shape[1] > cropped.shape[0]:
            #     cropped = cv2.rotate(cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)
            resized = image_resize(cropped)
            cropped_list.append(resized)

    if args.prv is True:
        res = cv2.hconcat(cropped_list)
        cv2.imshow('result', res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif args.prv is not None:
        res = cv2.hconcat(cropped_list)
        cv2.imwrite(args.prv, res)

def ocr(args):
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    ontology = CaptionOntology({args.prt:args.cls})
    detection = Detection(ontology, args.bth, args.tth)

    image = cv2.imread(args.img)
    result = detection.predict(image)

    if args.flt > 0:
    # if detection is in another detection, remove one 
        for i in range(len(result.xyxy)):
            for j in range(len(result.xyxy)):
                if i != j:
                    if (result.xyxy[i][0] >= result.xyxy[j][0] 
                        and result.xyxy[i][1] >= result.xyxy[j][1] 
                        and result.xyxy[i][2] <= result.xyxy[j][2] 
                        and result.xyxy[i][3] <= result.xyxy[j][3]):
                        if args.flt == 1: # keep outter box
                            result.xyxy[i] = [0, 0, 0, 0]
                        else: # keep inner box
                            result.xyxy[j] = [0, 0, 0, 0]
    

    cropped_list = []
    for i in range(len(result.xyxy)):
        # if area of the polygon is too small(eg. 100x100), skip
        area = (result.xyxy[i][2] - result.xyxy[i][0]) * (result.xyxy[i][3] - result.xyxy[i][1])
        if area < 10000:
            continue

        box = result.xyxy[i]
        mask = result.mask[i]

        # crop with oriented bounding box
        polygon =  mask_to_polygon(mask=mask)
        rotated_rect = cv2.minAreaRect(polygon)
        box = cv2.boxPoints(rotated_rect)
        box = np.intp(box).astype(int)

        # rotate the origin image
        rotated = cv2.warpAffine(image, cv2.getRotationMatrix2D(
            rotated_rect[0], rotated_rect[2], 1.0), 
            (image.shape[1], image.shape[0]))

        # crop the obb
        cropped = cv2.getRectSubPix(rotated, 
            (int(rotated_rect[1][0]), int(rotated_rect[1][1])), 
            rotated_rect[0])
        
        # if cropped width is larger than height, rotate the image 90 degree
        if args.rot and cropped.shape[1] > cropped.shape[0]:
            cropped = cv2.rotate(cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        ocr_result = ocr.ocr(cropped, cls=True)[0]
        boxes = [line[0] for line in ocr_result]
        texts = [line[1][0] for line in ocr_result]
        box_num = len(boxes)
        for i in range(box_num):
            box = np.reshape(np.array(boxes[i]), [-1, 1, 2]).astype(np.int64)
            cropped = cv2.polylines(np.array(cropped), [box], True, (0, 0, 255), 2)

        # put text on the image
        for i in range(box_num):
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1
            thickness = 2
            cropped = cv2.putText(cropped, texts[i], org, font, 
                   fontScale, (0, 0, 200), thickness, cv2.LINE_AA)
            print(texts[i])

        resized = image_resize(cropped, (1024, 1024))
        cropped_list.append(resized)

    if args.prv is True:
        res = cv2.hconcat(cropped_list)
        cv2.imshow('result', res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif args.prv is not None:
        res = cv2.hconcat(cropped_list)
        cv2.imwrite(args.prv, res)

def main():
    """
    Main function to parse command line arguments and perform object detection using GroundingDINO and SAM.
    This function sets up an argument parser to handle the following sub-command:
    - detect: Detect objects in an image based on provided prompts and classes.
    The 'detect' sub-command accepts the following arguments:
    - --image, -i: Path to the image (required).
    - --prt, -p: Prompt to detect (required).
    - --cls, -c: Class to detect (required).
    - --bth, -bt: Box threshold (optional, default is 0.35).
    - --tth, -tt: Text threshold (optional, default is 0.25).
    After parsing the arguments, the function initializes a CaptionOntology and Detection object,
    performs the detection on the provided image, and prints the result.
    """

    # read command line
    parser = argparse.ArgumentParser(
        prog="groundedsam_detection",
        description="Using GroundingDINO and SAM to detect objects in an image",
        ) 

    # add subparsers
    subparsers = parser.add_subparsers(help="detection sub-command help")

    # add detect subparser
    parser_detect = subparsers.add_parser("detect", help="detect objects in an image") 
    parser_detect.add_argument("-img", help="path to the image", required=True)

    parser_detect.add_argument("-prt", help="prompt to detect", required=True) 
    parser_detect.add_argument("-cls", help="class to detect", required=True) 

    parser_detect.add_argument("-bth", help="box threshold", type=float, default=0.35)
    parser_detect.add_argument("-tth", help="text threshold", type=float, default=0.25)

    parser_detect.add_argument("-flt", 
                               help=("filter out a box in another box, 0 = no filter, 1 = keep outter box, 2 = keep inner box"), 
                               type=int, default=0, choices=[0, 1, 2])

    parser_detect.add_argument("-dbb", help="draw bounding box", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser_detect.add_argument("-dmk", help="draw segments mask", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser_detect.add_argument("-dbd", help="draw segments border", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser_detect.add_argument("-obb", help="draw oriented bounding box", type=bool, default=False, action=argparse.BooleanOptionalAction)

    # if preview is true, show the detection result
    # if preview is a path, save the detection result to the path
    parser_detect.add_argument("-prv", help="preview the detection result", default=True)
    parser_detect.set_defaults(func=detect)


    # add crop subparser
    parser_crop = subparsers.add_parser("crop", help="crop the detected objects in ONE image") 
    parser_crop.add_argument("-img", help="path to the image", required=True)

    parser_crop.add_argument("-prt", help="prompt to detect", required=True) 
    parser_crop.add_argument("-cls", help="class to detect", required=True) 

    parser_crop.add_argument("-bth", help="box threshold", type=float, default=0.35)
    parser_crop.add_argument("-tth", help="text threshold", type=float, default=0.25)

    parser_crop.add_argument("-flt", 
                               help=("filter out a box in another box, 0 = no filter, 1 = keep outter box, 2 = keep inner box"), 
                               type=int, default=0, choices=[0, 1, 2])

    parser_crop.add_argument("-rot", help="using oriented bounding box to crop the image", default=False, type=bool, action=argparse.BooleanOptionalAction)
    # if preview is true, show the crop result
    # if preview is a path, save the crop result to the path
    parser_crop.add_argument("-prv", help="preview the crop result", default=True)
    parser_crop.set_defaults(func=crop)


    # add ocr subparser
    parser_ocr = subparsers.add_parser("ocr", help="ocr the detected objects") 
    parser_ocr.add_argument("-img", help="path to the image", required=True)

    parser_ocr.add_argument("-prt", help="prompt to detect", required=True) 
    parser_ocr.add_argument("-cls", help="class to detect", required=True) 

    parser_ocr.add_argument("-bth", help="box threshold", type=float, default=0.35)
    parser_ocr.add_argument("-tth", help="text threshold", type=float, default=0.25)
    parser_ocr.add_argument("-rot", help="rotate the cropped image for a better result", default=False, type=bool, action=argparse.BooleanOptionalAction)

    parser_ocr.add_argument("-flt", 
                            help=("filter out a box in another box, 0 = no filter, 1 = keep outter box, 2 = keep inner box"), 
                            type=int, default=0, choices=[0, 1, 2])
    parser_ocr.add_argument("-prv", help="preview the ocr result", default=True)
    parser_ocr.set_defaults(func=ocr)

    args = parser.parse_args()
    args.func(args)
            
if __name__ == "__main__":
    main()
