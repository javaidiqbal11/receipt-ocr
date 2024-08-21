from flask import Flask, request, jsonify
import cv2
import numpy as np
import torch
from yolov5 import YOLOv5
from gpt4_ocr import gpt4_ocr

app = Flask(__name__)

# Load models
yolov5_model = YOLOv5('../models/yolov5_receipt_detection.pt')

def extract_text_from_image(image):
    # Simulate text extraction from image to pass to GPT-4
    image_text = "sample image text"
    return gpt4_ocr(image_text)

@app.route('/process_receipt', methods=['POST'])
def process_receipt():
    file = request.files['image'].read()
    npimg = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    detections = yolov5_model.predict(img)
    
    for x1, y1, x2, y2, conf, cls in detections:
        cropped_receipt = img[y1:y2, x1:x2]
        break
    
    extracted_text = extract_text_from_image(cropped_receipt)
    
    return jsonify({'text': extracted_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
