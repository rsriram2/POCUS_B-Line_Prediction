import os
import glob
import torch
import torchvision
import torch.nn as nn
import scipy.io
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

# Load the entire model (architecture and weights)
def load_resnet18_3d(model_path: str) -> nn.Module:
    model = torch.load(model_path)  # Directly load the model
    return model

# Custom Dataset to handle .mat files
class VolumeDataset(Dataset):
    def __init__(self, mat_file_paths, transform=None):
        self.mat_file_paths = mat_file_paths
        self.transform = transform

        # Assign labels based on folder name (0 for non-b-line, 1 for b-line)
        self.labels = []
        for path in mat_file_paths:
            folder_name = os.path.basename(os.path.dirname(path))
            label = 1 if folder_name == 'b-line' else 0
            self.labels.append(label)

    def __len__(self):
        return len(self.mat_file_paths)

    def __getitem__(self, idx):
        mat_file_path = self.mat_file_paths[idx]
        mat_data = scipy.io.loadmat(mat_file_path)
        volume = mat_data['currentStack']

        if self.transform:
            volume = torch.stack([self.transform(volume[i]) for i in range(volume.shape[0])])

        label = self.labels[idx]
        return volume, label, mat_file_path  # Return file path as well

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert each slice to tensor
])

# Get validation file paths
val_mat_file_paths = glob.glob('/dcs05/ciprian/smart/pocus/rushil/stackedData/test/*/*.mat')
val_dataset = VolumeDataset(val_mat_file_paths, transform)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# Model initialization
model_path = '/dcs05/ciprian/smart/pocus/rushil/model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_resnet18_3d(model_path)
model = model.to(device)
model.eval()  # Set model to evaluation mode

# Initialize lists to store file paths, predictions, and actual labels
file_paths = []
predictions = []
actual_labels = []

with torch.no_grad():
    for volumes, labels, mat_file_paths in val_dataloader:
        volumes = volumes.to(device)
        labels = labels.to(device)
        volumes = volumes.permute(0, 2, 3, 4, 1)  # [batch, channels, height, width, depth]
        
        outputs = model(volumes)
        _, predicted = torch.max(outputs, 1)

        file_paths.extend(mat_file_paths)
        predictions.extend(predicted.cpu().numpy())
        actual_labels.extend(labels.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(actual_labels, predictions)
sensitivity = recall_score(actual_labels, predictions, pos_label=1)  # Sensitivity for positive class
specificity = recall_score(actual_labels, predictions, pos_label=0)  # Specificity for negative class

# Confusion Matrix
conf_matrix = confusion_matrix(actual_labels, predictions)
tn, fp, fn, tp = conf_matrix.ravel()

print(f"Accuracy: {accuracy:.4f}")
print(f"Sensitivity (Recall for class 1): {sensitivity:.4f}")
print(f"Specificity (Recall for class 0): {specificity:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Save predictions, actual labels, and file paths to CSV
df = pd.DataFrame({
    'File Path': file_paths,
    'Actual Label': actual_labels,
    'Prediction': predictions
})

csv_file_path = '/dcs05/ciprian/smart/pocus/rushil/predictions_with_labels.csv'
df.to_csv(csv_file_path, index=False)

print(f'Predictions with labels saved to {csv_file_path}')
