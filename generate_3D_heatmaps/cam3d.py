import sys
import os

# Add the path to the train_3D_models directory to the system path
sys.path.append('/dcs05/ciprian/smart/pocus/rushil/POCUS_B-Line_Prediction/train_3D_models')

# Now you can import the module
from architectures_by_others import UNet3D_Born_etal

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import scipy.io
import glob
from torchvision import transforms

class GradCAM3D:
    def __init__(self, model, target_layer):
        """
        Args:
            model (torch.nn.Module): The trained 3D model.
            target_layer (torch.nn.Module): The layer to visualize (e.g., the last conv layer).
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook to capture gradients and activations
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        """
        Generates Grad-CAM for a given input.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape [1, C, D, H, W].
            target_class (int, optional): Class index for which Grad-CAM is computed. If None, the class with the highest score is used.

        Returns:
            numpy.ndarray: Grad-CAM heatmap of shape [D, H, W].
        """
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()

        # Compute weights
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)  # Global average pooling over D, H, W

        # Compute Grad-CAM
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # Weighted sum of channels
        cam = F.relu(cam)  # Apply ReLU

        # Normalize the CAM
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # Normalize to [0, 1]

        return cam

    def save_cam(self, cam, save_path):
        """
        Saves a slice of the Grad-CAM heatmap as an image.

        Args:
            cam (numpy.ndarray): Grad-CAM heatmap of shape [D, H, W].
            save_path (str): Path to save the heatmap image.
        """
        # Select the middle slice along the depth dimension
        mid_slice = cam[cam.shape[0] // 2, :, :]

        # Save the heatmap
        plt.imshow(mid_slice, cmap='jet')
        plt.colorbar()
        plt.title("Grad-CAM Heatmap")
        plt.savefig(save_path)
        plt.close()

# Dataset class for loading the 3D volumes
class VolumeDataset(Dataset):
    def __init__(self, mat_file_paths, transform=None):
        # Store file paths and transformations
        self.mat_file_paths = mat_file_paths
        self.transform = transform

        # Assign labels based on folder name
        self.labels = []
        self.file_names = []  # Store the file names

        for path in mat_file_paths:
            folder_name = os.path.basename(os.path.dirname(path))
            label = 1 if folder_name == 'b-line' else 0
            self.labels.append(label)
            self.file_names.append(os.path.basename(path))  # Store the file name

    def __len__(self):
        return len(self.mat_file_paths)

    def __getitem__(self, idx):
        mat_file_path = self.mat_file_paths[idx]
        mat_data = scipy.io.loadmat(mat_file_path)
        volume = mat_data['currentStack']

        if self.transform:
            volume = torch.stack([self.transform(volume[i]) for i in range(volume.shape[0])])

        label = self.labels[idx]
        file_name = self.file_names[idx]  # Extract the file name as a string
        return volume, label, file_name  # Return the file name

# Transformation to convert each slice to a tensor
transform = transforms.Compose([
    transforms.ToTensor(),          # Convert each slice to tensor (this converts to [0, 1] range)
])

# Load validation data
val_mat_file_paths = glob.glob('/dcs05/ciprian/smart/pocus/rushil/3D_data/masked/validation/*/*.mat')
val_dataset = VolumeDataset(val_mat_file_paths, transform)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Load the model
model = UNet3D_Born_etal(pretrained=False, input_channels=3, N_classes=2)

# Load custom weights with mapping to CPU
model_weights_path = '/dcs05/ciprian/smart/pocus/rushil/all_model_paths/masked_model_paths/unet.pth'
model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
target_layer = model.encoder.down_tr256

grad_cam = GradCAM3D(model, target_layer)

# Define device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  # Move the model to the device

output_dir = 'gradcam_outputs'
os.makedirs(output_dir, exist_ok=True)

for i, (input_volume, label, file_name_tuple) in enumerate(val_dataloader):
     file_name = file_name_tuple[0]
     # Print the file_name to debug
     print(f"File Name: {file_name}, Type: {type(file_name)}")
     # Permute the dimensions of input_volume before passing to Grad-CAM
     input_volume = input_volume.to(device)  # Send to GPU if available
     input_volume = input_volume.permute(0, 2, 1, 3, 4)  # Ensure correct dimensions
     label = label.to(device)
     heatmap = grad_cam.generate_cam(input_volume, target_class=label.item())
     save_path = os.path.join(output_dir, file_name.replace('.mat', '_heatmap.png'))  # Use the same file name
     grad_cam.save_cam(heatmap, save_path)
