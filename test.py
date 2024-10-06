import cv2
from pyzbar.pyzbar import decode, ZBarSymbol
from cv2.barcode import BarcodeDetector

image = cv2.imread("rotated_0.jpg")
converted = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# res = decode(converted, symbols=[ZBarSymbol.CODE128])
# print(res)

bd = BarcodeDetector('.cache/wechatcv/sr.prototxt', '.cache/wechatcv/sr.caffemodel')
detected, corners = bd.detect(converted)
if not detected:
    print("No barcode detected")
else:
    print("Barcode detected")
    for corner in corners:
        rect = cv2.boundingRect(corner)

        # crop the barcode
        cropped = image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        res = decode(cropped, symbols=[ZBarSymbol.CODE128])
        print(res)
