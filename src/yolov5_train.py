from yolov5 import YOLOv5

def train_yolov5_model():
    yolov5_model = YOLOv5('yolov5s.pt')  # Pre-trained YOLOv5 model
    yolov5_model.train(data='../data/annotated_data/data.yaml', epochs=50, batch_size=16)
    yolov5_model.save('../models/yolov5_receipt_detection.pt')

if __name__ == "__main__":
    train_yolov5_model()
