import os
import glob
import torch
torch.cuda.empty_cache()
import torchvision
import torch.nn as nn
import scipy.io
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

# Load the pre-trained ResNet18 model for 3D
def load_resnet18_3d(input_channels: int, N_classes: int, model_path: str) -> nn.Module:
    model = torchvision.models.video.r3d_18(pretrained=True)
    model.fc = nn.Linear(in_features=512, out_features=N_classes)
    model.load_state_dict(torch.load(model_path))
    return model

# Dataset class for loading the 3D volumes
class VolumeDataset(Dataset):
    def __init__(self, mat_file_paths, transform=None):
        # Store file paths and transformations
        self.mat_file_paths = mat_file_paths
        self.transform = transform

        # Assign labels based on folder name
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
        return volume, label

# Transformation to convert each slice to a tensor
transform = transforms.Compose([
    transforms.ToTensor(),          # Convert each slice to tensor (this converts to [0, 1] range)
])

# Load training data
train_mat_file_paths = glob.glob('/dcs05/ciprian/smart/pocus/rushil/masked_stackedData/training/*/*.mat')
train_dataset = VolumeDataset(train_mat_file_paths, transform)
train_dataloader = DataLoader(train_dataset, batch_size=18, shuffle=False)

# Load validation data
val_mat_file_paths = glob.glob('/dcs05/ciprian/smart/pocus/rushil/masked_stackedData/validation/*/*.mat')
val_dataset = VolumeDataset(val_mat_file_paths, transform)
val_dataloader = DataLoader(val_dataset, batch_size=18, shuffle=False)

# Model initialization
input_channels = 3
N_classes = 2
model_path = '/dcs05/ciprian/smart/pocus/rushil/video_resnet18_3d.pth'  # Path to your pre-trained model
model = load_resnet18_3d(input_channels, N_classes, model_path)
model.train()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  # For binary classification, make sure labels are 0 and 1
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = model.to(device)
criterion = criterion.to(device)

# Training loop with multiple epochs
  # Set the number of epochs you want to train for

def validate(model, val_dataloader):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # No need to compute gradients during validation
        for volumes, labels in val_dataloader:
            volumes = volumes.to(device)
            labels = labels.to(device)
            volumes = volumes.permute(0, 2, 3, 4, 1)  # [batch, channels, height, width, depth]
            outputs = model(volumes)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_val_loss = val_loss / len(val_dataloader)
    accuracy = 100 * correct / total
    print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return avg_val_loss, accuracy

# Start training
num_epochs = 10
best_val_loss = float('inf')
writer = SummaryWriter(log_dir='/dcs05/ciprian/smart/pocus/rushil/masked_model_logs')
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    model.train()  # Set the model to training mode
    running_loss = 0.0

    for batch_idx, (volumes, labels) in enumerate(train_dataloader):
        volumes = volumes.to(device)  # Move data to device
        labels = labels.to(device)
        
        volumes = volumes.permute(0, 2, 3, 4, 1)  # [batch, channels, height, width, depth]

        # Forward pass
        optimizer.zero_grad()  # Zero the gradient buffers
        outputs = model(volumes)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights

        running_loss += loss.item()
        writer.add_scalar('Training Loss/Batch', loss.item(), epoch * len(train_dataloader) + batch_idx)

        if batch_idx % 1 == 0:
            print(f"Batch {batch_idx + 1}, Loss: {loss.item()}")

    avg_train_loss = running_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1} Training Loss: {avg_train_loss:.4f}")

     # Log epoch training loss
    writer.add_scalar('Training Loss/Epoch', avg_train_loss, epoch + 1)

    # Validate after each epoch
    avg_val_loss, val_accuracy = validate(model, val_dataloader)
    
    # Log validation metrics
    writer.add_scalar('Validation Loss/Epoch', avg_val_loss, epoch + 1)
    writer.add_scalar('Validation Accuracy/Epoch', val_accuracy, epoch + 1)

     # Save model based on best validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), '/dcs05/ciprian/smart/pocus/rushil/masked_model_paths/best_model_e10.pth')
        print(f"Best model saved with validation loss: {best_val_loss:.4f}")

writer.close()
