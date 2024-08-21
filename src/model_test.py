import cv2
import torch
from ultralytics import YOLO
from gpt4_ocr import gpt4_ocr
from data_loader import load_images_from_folder

def test_yolov8_model(yolov8_model, test_image_path):
    img = cv2.imread(test_image_path)
    results = yolov8_model(img)  # Perform detection
    for detection in results.xyxy[0]:  # Assuming single image and accessing results
        x1, y1, x2, y2, conf, cls = map(int, detection[:6])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cropped_receipt = img[y1:y2, x1:x2]
        return cropped_receipt

def extract_text_from_image(image):
    # Simulate text extraction from image to pass to GPT-4
    image_text = "sample image text"
    return gpt4_ocr(image_text)

if __name__ == "__main__":
    yolov8_model = YOLO('../models/yolov8_receipt_detection.pt')
    test_image_path = '../data/test/test_image.jpg'
    
    cropped_receipt_image = test_yolov8_model(yolov8_model, test_image_path)
    
    extracted_text = extract_text_from_image(cropped_receipt_image)
    print("Extracted Text:", extracted_text)
