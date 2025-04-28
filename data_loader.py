import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import random

class RotationDataset(Dataset):
    def __init__(self, root_dir, transform=None, rotation_range=(-45, 45)):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
            rotation_range (tuple): Range of random rotation angles in degrees.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.rotation_range = rotation_range
        
        # Get all image file paths
        self.image_paths = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply initial transforms if provided (like resize)
        if self.transform:
            image = self.transform(image)
        
        # Apply random rotation
        rotation_angle = random.uniform(self.rotation_range[0], self.rotation_range[1])
        rotated_image = transforms.functional.rotate(image, rotation_angle)
        
        # The target is the negative of the applied rotation (to correct it)
        target_angle = -rotation_angle
        
        # Normalize the target angle to be between -1 and 1 for better training
        normalized_target = target_angle / 90.0  # Assuming max rotation is ±90 degrees
        
        return rotated_image, torch.tensor([normalized_target], dtype=torch.float32)

def get_data_loaders(data_dir, batch_size=32, img_size=224, rotation_range=(-45, 45)):
    """
    Create training and validation data loaders
    
    Args:
        data_dir (str): Path to the dataset directory
        batch_size (int): Batch size for training
        img_size (int): Size to resize images to
        rotation_range (tuple): Range of random rotation angles
        
    Returns:
        train_loader, val_loader: DataLoader objects for training and validation
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = RotationDataset(root_dir=data_dir, transform=transform, rotation_range=rotation_range)
    
    # Split dataset into training and validation sets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader
