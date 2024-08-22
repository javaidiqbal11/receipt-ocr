from ultralytics import YOLO
import os

from ultralytics.data.explorer.explorer import Explorer


def train_yolov8_model(data_dir, model_save_path):
    # Load YOLOv8 model (either a pre-trained one or a custom architecture)
    model = YOLO('yolov8n.yaml')  # Load a YOLOv8 nano model configuration for training
    
    # Set the training parameters
    model.train(data=data_dir, epochs=50, imgsz=640, batch=16, name='yolov8_receipt_detection')
    
    # Save the trained model
    model.export(format="torchscript", path=model_save_path)

if __name__ == "__main__":
    data_dir = {
        'train': '../data/images',  # Path to training images
        'val': '../data/images',    # Path to validation images (can be the same as train if split not provided)
        'names': '../data/classes.txt',  # Path to the classes file
        'labels': '../data/labels'  # Path to the labels
    }
    
    model_save_path = '../models/yolov8_receipt_detection.pt'
    train_yolov8_model(data_dir, model_save_path)
