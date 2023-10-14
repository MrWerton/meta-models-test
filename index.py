from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw, ImageFont
import requests
import os

directory_path = os.path.join(".", "woman.jpg")

image = Image.open(directory_path)

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]


font = ImageFont.load_default()
draw = ImageDraw.Draw(image)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    class_name = model.config.id2label[label.item()]
    confidence = round(score.item(), 3)
    
    
    draw.rectangle(box, outline="cyan", width=3)

    
    draw.text((box[0], box[1] - 20), class_name, fill="red", font=font)


image.save("detected_image.jpg")


image.show()
