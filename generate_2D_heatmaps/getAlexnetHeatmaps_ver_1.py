import os
import torch
import torchvision.transforms as transforms
from torchvision.models import alexnet, AlexNet_Weights
from PIL import Image
import numpy as np
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt

# Load pretrained AlexNet model
model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 2)

model_path = '/dcs05/ciprian/smart/pocus/rushil/model_paths/alexnet/ultraAlexNet_model.pth'
model.load_state_dict(torch.load(model_path))
input_dir = '/dcs05/ciprian/smart/pocus/data/separated_masked_png/test'
output_dir = '/dcs05/ciprian/smart/pocus/rushil/Heatmaps/alexnet/version1/'

print(model)
os.makedirs(output_dir, exist_ok=True)

preprocess = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model.eval()  # Set model to evaluation mode

for foldername in os.listdir(input_dir):
    folder_path = os.path.join(input_dir, foldername)
    if os.path.isdir(folder_path):
        output_subdir = os.path.join(output_dir, foldername.lower())
        os.makedirs(output_subdir, exist_ok=True)
        for filename in os.listdir(folder_path):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                img_path = os.path.join(folder_path, filename)
                img = Image.open(img_path).convert('RGB')
                img_tensor = preprocess(img).unsqueeze(0)
                with torch.no_grad():
                    activations = model.features(img_tensor)
                heatmap = torch.mean(activations, dim=1).squeeze().cpu()
                heatmap = np.maximum(heatmap, 0)
                heatmap /= torch.max(heatmap)
                heatmap = heatmap.numpy()
                heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
                overlayed_img = np.array(img)
                heatmap_normalized = (heatmap * 255).astype(np.uint8)
                heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
                overlayed_img_with_heatmap = cv2.addWeighted(overlayed_img, 0.5, heatmap_colored, 0.5, 0)
                base_filename = os.path.splitext(filename)[0]
                overlayed_img_save_path = os.path.join(output_subdir, f"{base_filename}.png")
                cv2.imwrite(overlayed_img_save_path, overlayed_img_with_heatmap)

                print(f"Generated and saved heatmap and overlay for {filename} in {foldername}")

print("Heatmap and overlay generation and saving completed.")
