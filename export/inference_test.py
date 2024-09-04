import os
import time
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import numpy as np

class ObjectDetection:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self, image_path):
        results = self.model.predict(image_path)
        return results[0]

    def display_box_info(self, box, result):
        cords = [round(x) for x in box.xyxy[0].tolist()]
        class_id = result.names[box.cls[0].item()]
        conf = round(box.conf[0].item(), 2)
        print(f"Object type: {class_id}")
        print(f"Coordinates: {cords}")
        print(f"Probability: {conf}")
        print("---")

    def display_results(self, result):
        for box in result.boxes:
            self.display_box_info(box, result)

    def visualize_detections(self, image_path, result):
        image = Image.open(image_path)
        image_np = np.array(image)
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image_np)

        for box in result.boxes:
            cords = [round(x) for x in box.xyxy[0].tolist()]
            rect = patches.Rectangle(
                (cords[0], cords[1]), cords[2] - cords[0], cords[3] - cords[1],
                linewidth=2, edgecolor='yellow', facecolor='none'
            )
            ax.add_patch(rect)
            label = f"{result.names[box.cls[0].item()]} ({round(box.conf[0].item(), 2)})"
            ax.text(cords[0], cords[1] - 10, label, color='black', fontsize=10, bbox=dict(facecolor='yellow', alpha=1.0))
        
        ax.axis('off')
        plt.show()

class CustomImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.transform = transform
        self.image_paths = [
            os.path.join(directory, filename)
            for filename in os.listdir(directory)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def run_inference(model, data_loader):
    all_predictions = []
    start_time = time.time()

    with torch.no_grad():
        for batch_images in data_loader:
            batch_images = batch_images.to('cuda')
            outputs = model(batch_images)

    end_time = time.time()
    print(f"Inference time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    # Initialize model
    model_path = "yolov8s.engine"
    image_path = "img/traffic.jpg"
    od = ObjectDetection(model_path)

    # Predict and display results
    result = od.predict(image_path)
    od.display_results(result)
    od.visualize_detections(image_path, result)

    # Setup dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])

    image_directory = "images"
    dataset = CustomImageDataset(directory=image_directory, transform=transform)
    batch_size = 8
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Run inference
    run_inference(od.model, data_loader)
