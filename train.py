import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

def train_model(model, train_loader, val_loader, device, num_epochs=20, learning_rate=0.001, 
                save_dir='./checkpoints'):
    """
    Train the rotation correction model
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on ('cuda' or 'cpu')
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for optimizer
        save_dir: Directory to save model checkpoints
        
    Returns:
        model: The trained model
        history: Dictionary containing training history
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Loss function and optimizer
    # criterion = nn.MSELoss()
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    # l1_loss = nn.L1Loss()  # L1 Loss
    # smooth_l1_loss = nn.SmoothL1Loss()



    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    # Track best validation loss and accuracy
    best_val_loss = float('inf')
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': []  # Mean Absolute Error in degrees
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, angles in train_bar:
            images, angles = images.to(device), angles.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # # Calculate loss
            loss = criterion(outputs, angles)

            # Calculate loss (Hybrid loss: MSE + 0.5 * L1)
            # loss = criterion(outputs, angles) + 0.5 * l1_loss(outputs, angles)
            # Calculate smooth loss
            # loss = smooth_l1_loss(outputs, angles)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * images.size(0)
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        # Calculate average training loss for the epoch
        train_loss = train_loss / len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        angle_errors = []
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for images, angles in val_bar:
                images, angles = images.to(device), angles.to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Calculate loss
                loss = criterion(outputs, angles)
                # loss = criterion(outputs, angles) + 0.5 * l1_loss(outputs, angles)
                # Calculate smooth loss
                # loss = smooth_l1_loss(outputs, angles)
                val_loss += loss.item() * images.size(0)
                
                # Convert normalized angle predictions back to degrees
                pred_angles_deg = outputs.cpu().numpy() * 90.0
                true_angles_deg = angles.cpu().numpy() * 90.0
                
                # Calculate absolute error in degrees
                batch_errors = np.abs(pred_angles_deg - true_angles_deg)
                angle_errors.extend(batch_errors.flatten().tolist())
                
                val_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        # Calculate average validation loss and MAE for the epoch
        val_loss = val_loss / len(val_loader.dataset)
        val_mae = np.mean(angle_errors)
        
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val MAE: {val_mae:.2f}°")
        
        # Learning rate scheduler step
        scheduler.step(val_loss)
        
        # Save model if it has the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"Saved best model with validation loss: {val_loss:.4f}")
        
        # Save model checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    return model, history

def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on test data
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        device: Device to evaluate on
        
    Returns:
        test_mae: Mean Absolute Error in degrees
        predictions: List of (true_angle, predicted_angle) pairs
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for images, angles in tqdm(test_loader, desc="Evaluating"):
            images, angles = images.to(device), angles.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Convert normalized angle predictions back to degrees
            pred_angles_deg = outputs.cpu().numpy() * 90.0
            true_angles_deg = angles.cpu().numpy() * 90.0
            
            # Store predictions
            for i in range(len(images)):
                predictions.append((true_angles_deg[i, 0], pred_angles_deg[i, 0]))
    
    # Calculate mean absolute error
    true_angles, pred_angles = zip(*predictions)
    test_mae = np.mean(np.abs(np.array(true_angles) - np.array(pred_angles)))
    
    print(f"Test Mean Absolute Error: {test_mae:.2f}°")
    
    return test_mae, predictions

def plot_training_history(history, save_path=None):
    """
    Plot training and validation metrics
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history['val_mae'], label='Validation MAE')
    plt.title('Mean Absolute Error (degrees)')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (°)')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_prediction_results(predictions, num_samples=10, save_path=None):
    """
    Plot true vs predicted angles for a sample of test images
    
    Args:
        predictions: List of (true_angle, predicted_angle) pairs
        num_samples: Number of samples to plot
        save_path: Path to save the plot
    """
    # Select a random sample of predictions
    indices = np.random.choice(len(predictions), min(num_samples, len(predictions)), replace=False)
    samples = [predictions[i] for i in indices]
    
    true_angles, pred_angles = zip(*samples)
    errors = np.array(pred_angles) - np.array(true_angles)
    
    # Create a figure
    plt.figure(figsize=(12, 6))
    
    # Bar plot of true vs predicted angles
    plt.subplot(1, 2, 1)
    x = np.arange(len(samples))
    width = 0.35
    plt.bar(x - width/2, true_angles, width, label='True Angle')
    plt.bar(x + width/2, pred_angles, width, label='Predicted Angle')
    plt.xlabel('Sample')
    plt.ylabel('Angle (degrees)')
    plt.title('True vs Predicted Angles')
    plt.xticks(x, [f"{i+1}" for i in range(len(samples))])
    plt.legend()
    
    # Error histogram
    plt.subplot(1, 2, 2)
    all_errors = np.array([p[1] - p[0] for p in predictions])
    plt.hist(all_errors, bins=20)
    plt.xlabel('Error (degrees)')
    plt.ylabel('Count')
    plt.title(f'Error Distribution (MAE: {np.mean(np.abs(all_errors)):.2f}°)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
