import cv2
import torch
from yolov5 import YOLOv5
from gpt4_ocr import gpt4_ocr
from data_loader import load_images_from_folder

def test_yolov5_model(yolov5_model):
    test_images = load_images_from_folder('../data/test')
    detections = [yolov5_model.predict(img) for img in test_images]

    for img, det in zip(test_images, detections):
        for x1, y1, x2, y2, conf, cls in det:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cropped_receipt = img[y1:y2, x1:x2]
            return cropped_receipt

def extract_text_from_image(image):
    # Here you would convert the image to text, but for the GPT-4 API, you simulate this step
    # Assume 'image_text' is the result of image processing before sending to GPT-4
    image_text = "sample image text"
    return gpt4_ocr(image_text)

if __name__ == "__main__":
    yolov5_model = YOLOv5('../models/yolov5_receipt_detection.pt')
    cropped_receipt_image = test_yolov5_model(yolov5_model)
    
    extracted_text = extract_text_from_image(cropped_receipt_image)
    print("Extracted Text:", extracted_text)
