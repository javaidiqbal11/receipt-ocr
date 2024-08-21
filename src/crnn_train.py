import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from crnn_model import CRNN  # Assuming CRNN model defined in another file

def train_crnn_model():
    ocr_model = CRNN()
    criterion = nn.CTCLoss()
    optimizer = optim.Adam(ocr_model.parameters(), lr=0.001)
    train_loader = DataLoader('../data/ocr_dataset', batch_size=32, shuffle=True)

    for epoch in range(10):
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = ocr_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/10, Loss: {loss.item()}')

    torch.save(ocr_model.state_dict(), '../models/crnn_ocr_model.pth')

if __name__ == "__main__":
    train_crnn_model()
