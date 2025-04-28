import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image

def denormalize_image(tensor):
    """Convert a normalized image tensor back to a displayable image"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Convert to numpy and transpose from (C,H,W) to (H,W,C)
    img = tensor.cpu().numpy().transpose(1, 2, 0)

    # Denormalize
    img = std * img + mean
    img = np.clip(img, 0, 1)

    return img

def visualize_rotation_correction(model, image_path, device, img_size=224):
    """
    Visualize how the model corrects a rotated image

    Args:
        model: The trained rotation correction model
        image_path: Path to the image to visualize
        device: Device to run the model on
        img_size: Image size for the model
    """
    # Load image
    original_image = Image.open(image_path).convert('RGB')

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create randomly rotated versions
    rotation_angles = [-45, -30, -15, 0, 15, 30, 45]
    rotated_images = []
    corrected_images = []
    predicted_angles = []

    for angle in rotation_angles:
        # Rotate image
        rotated_img = transforms.functional.rotate(original_image, angle)

        # Transform for model input
        input_tensor = transform(rotated_img).unsqueeze(0).to(device)

        # Get prediction
        model.eval()
        with torch.no_grad():
            pred_angle_normalized = model(input_tensor).item()

        # Convert normalized prediction to degrees
        pred_angle_degrees = pred_angle_normalized * 90.0
        predicted_angles.append(pred_angle_degrees)

        # Create corrected image using the predicted angle
        corrected_img = transforms.functional.rotate(rotated_img, pred_angle_degrees)

        rotated_images.append(rotated_img)
        corrected_images.append(corrected_img)

    # Visualize results
    fig, axes = plt.subplots(3, len(rotation_angles), figsize=(20, 10))

    # Original rotated images
    for i, (angle, img) in enumerate(zip(rotation_angles, rotated_images)):
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Rotated: {angle}°")
        axes[0, i].axis('off')

    # Corrected images
    for i, (img, pred_angle) in enumerate(zip(corrected_images, predicted_angles)):
        axes[1, i].imshow(img)
        axes[1, i].set_title(f"Corrected: {pred_angle:.1f}°")
        axes[1, i].axis('off')

    # Error visualization
    for i, (original_angle, pred_angle) in enumerate(zip(rotation_angles, predicted_angles)):
        error = pred_angle + original_angle  # Should be close to zero for perfect correction
        color = 'green' if abs(error) < 5 else 'red'
        axes[2, i].bar([0], [error], color=color)
        axes[2, i].set_ylim(-20, 20)
        axes[2, i].set_title(f"Error: {error:.1f}°")
        axes[2, i].set_xticks([])
        axes[2, i].axhline(y=0, color='k', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.show()

    return rotation_angles, predicted_angles

def visualize_predictions_batch(model, data_loader, device, num_samples=8):
    """
    Visualize model predictions on a batch of images

    Args:
        model: The trained rotation correction model
        data_loader: DataLoader with rotated images
        device: Device to run the model on
        num_samples: Number of samples to visualize
    """
    # Get a batch from the data loader
    model.eval()
    images, angles = next(iter(data_loader))

    # Limit to the requested number of samples
    images = images[:num_samples]
    angles = angles[:num_samples]

    # Get predictions
    with torch.no_grad():
        outputs = model(images.to(device))

    # Convert normalized angles back to degrees
    true_angles_deg = angles.numpy() * 90.0
    pred_angles_deg = outputs.cpu().numpy() * 90.0

    # Prepare figure
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 8))

    # Original rotated images
    for i in range(num_samples):
        img = denormalize_image(images[i])
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"True: {true_angles_deg[i][0]:.1f}°")
        axes[0, i].axis('off')

    # Visual representation of the prediction error
    for i in range(num_samples):
        error = pred_angles_deg[i][0] - true_angles_deg[i][0]
        color = 'green' if abs(error) < 5 else 'red'
        axes[1, i].bar([0], [pred_angles_deg[i][0]], color=color)
        axes[1, i].bar([1], [true_angles_deg[i][0]], color='blue', alpha=0.5)
        axes[1, i].set_title(f"Pred: {pred_angles_deg[i][0]:.1f}°\nError: {error:.1f}°")
        axes[1, i].set_xticks([0, 1])
        axes[1, i].set_xticklabels(['Pred', 'True'])
        axes[1, i].set_ylim(min(-90, min(pred_angles_deg.min(), true_angles_deg.min()) - 10),
                           max(90, max(pred_angles_deg.max(), true_angles_deg.max()) + 10))

    plt.tight_layout()
    plt.show()
