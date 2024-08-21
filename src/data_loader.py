import cv2
import matplotlib.pyplot as plt
import os

def load_images_from_folder(folder, num_images=5):
    images = []
    for i, filename in enumerate(os.listdir(folder)):
        if i >= num_images:
            break
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def visualize_images(images):
    for i, img in enumerate(images):
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f'Image {i+1}')
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    images = load_images_from_folder('../data/train')
    visualize_images(images)
