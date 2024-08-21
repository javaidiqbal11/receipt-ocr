from flask import Flask, request, jsonify
import cv2
import numpy as np
import torch
from yolov5 import YOLOv5
from crnn_model import CRNN  # Assuming CRNN model defined in another file

app = Flask(__name__)

# Load models
yolov5_model = YOLOv5('../models/yolov5_receipt_detection.pt')
ocr_model = CRNN()
ocr_model.load_state_dict(torch.load('../models/crnn_ocr_model.pth'))
ocr_model.eval()

def decode_output(output):
    # Placeholder for output decoding logic
    return 'Decoded Text'

@app.route('/process_receipt', methods=['POST'])
def process_receipt():
    file = request.files['image'].read()
    npimg = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    detections = yolov5_model.predict(img)
    
    for x1, y1, x2, y2, conf, cls in detections:
        cropped_receipt = img[y1:y2, x1:x2]
        break
    
    output = ocr_model(cropped_receipt)
    text = decode_output(output)
    
    return jsonify({'text': text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
