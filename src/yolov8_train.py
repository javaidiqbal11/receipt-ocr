from ultralytics import YOLO

def train_yolov8_model(data_yaml, model_save_path):
    # Load YOLOv8 model (either a pre-trained one or a custom architecture)
    model = YOLO('yolov8m.pt')  # Load a YOLOv8 medium pre-trained model (recommended for training)
    
    # Set the training parameters
    model.train(data=data_yaml, epochs=50, imgsz=1024, batch=16, name='yolov8_receipt_detection')
    
    # Save the trained model
    model.export(format="torchscript", path=model_save_path)

if __name__ == "__main__":
    # Path to the dataset YAML file
    data_yaml = 'data/dataset.yaml'  # Ensure this points to your dataset.yaml file
    
    # Path where the trained model will be saved
    model_save_path = 'models/yolov8_receipt_detection.pt'
    
    # Train the YOLOv8 model
    train_yolov8_model(data_yaml, model_save_path)

# follow the ultralytics documentation
