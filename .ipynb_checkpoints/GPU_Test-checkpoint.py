import torch
import torchvision.transforms as transforms
import onnx
import onnxruntime as ort
import cv2
import numpy as np
from PIL import Image, ImageDraw
from torch2trt import torch2trt
import time

class YOLOModel:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

    def infer(self, img):
        img = self.preprocess_image(img)
        num_runs = 100
        for _ in range(num_runs):
            st = time.perf_counter()
            outputs = self.session.run(None, {self.input_name: img})
            et = time.perf_counter() - st
            print(f"FPS {1/et} {et}")
        return outputs

    def preprocess_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, (self.input_shape[3], self.input_shape[2]))  # Resize to (640, 640)
        img = img.astype(np.float32) / 255.0  # Normalize the image
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = np.transpose(img, (0, 3, 1, 2))  # Reorder dimensions
        return img

class ImageProcessor:
    yolo_classes = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
        "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
        "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
        "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    def __init__(self, img_path):
        self.img_path = img_path

    def load_image(self):
        img = Image.open(self.img_path)
        img = img.resize((640, 640))
        return img

    def draw_boxes(self, boxes):
        img = Image.open(self.img_path)
        img = img.resize((640, 640))
        draw = ImageDraw.Draw(img)
        for box in boxes:
            x1, y1, x2, y2, class_id, prob = box
            draw.rectangle((int(x1), int(y1), int(x2), int(y2)), outline="#00ff00")
            draw.text((int(x1), int(y1)), f"{class_id} {prob:.2f}", fill="#00ff00")
        return img

    @staticmethod
    def parse_row(row):
        xc, yc, w, h = row[:4]
        x1 = (xc - w/2)
        y1 = (yc - h/2)
        x2 = (xc + w/2)
        y2 = (yc + h/2)
        prob = row[4:].max()
        class_id = row[4:].argmax()
        label = ImageProcessor.yolo_classes[class_id]
        return [x1, y1, x2, y2, label, prob]

    @staticmethod
    def filter_boxes(output, threshold=0.5):
        return [box for box in [ImageProcessor.parse_row(row) for row in output] if box[5] > threshold]

def main():
    model_path = 'int8_pre_yolov8s.onnx'
    img_path = 'bus.jpg'

    yolo_model = YOLOModel(model_path)
    image_processor = ImageProcessor(img_path)

    img = cv2.imread(img_path)
    outputs = yolo_model.infer(img)

    output = outputs[0]
    boxes = ImageProcessor.filter_boxes(output)

    processed_img = image_processor.draw_boxes(boxes)
    processed_img.show()

if __name__ == "__main__":
    main()
