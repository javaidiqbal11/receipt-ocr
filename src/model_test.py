import cv2
import torch
from yolov5 import YOLOv5
from crnn_model import CRNN  # Assuming CRNN model defined in another file
from data_loader import load_images_from_folder

def test_yolov5_model(yolov5_model):
    test_images = load_images_from_folder('../data/test')
    detections = [yolov5_model.predict(img) for img in test_images]

    for img, det in zip(test_images, detections):
        for x1, y1, x2, y2, conf, cls in det:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow('Detection', img)
        cv2.waitKey(0)

def test_crnn_model(ocr_model):
    test_image = cv2.imread('../data/test/test_image.jpg')
    output = ocr_model(test_image)
    text = decode_output(output)  # Custom function to decode model output
    print('Detected Text:', text)

if __name__ == "__main__":
    yolov5_model = YOLOv5('../models/yolov5_receipt_detection.pt')
    test_yolov5_model(yolov5_model)
    
    ocr_model = CRNN()
    ocr_model.load_state_dict(torch.load('../models/crnn_ocr_model.pth'))
    ocr_model.eval()
    test_crnn_model(ocr_model)
