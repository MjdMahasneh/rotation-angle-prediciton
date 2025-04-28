import torch
import argparse
import os
import random
import numpy as np
from data_loader import get_data_loaders
from model import get_model
from train import train_model, evaluate_model, plot_training_history, plot_prediction_results
from visualization import visualize_rotation_correction, visualize_predictions_batch

def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description='Train a rotation correction model')
    parser.add_argument('--data_dir', type=str, default='./dataset',
                        help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--rotation_range', type=str, default='-45,45',
                        help='Range of random rotations (min,max)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save models')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions', default=True)
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation', default=True)
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth', help='Path to a pretrained model')
    # parser.add_argument('--model_path', type=str, default=None, help='Path to a pretrained model')


    args = parser.parse_args()
    print("Arguments:", args)
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Parse rotation range
    rot_min, rot_max = map(int, args.rotation_range.split(','))
    rotation_range = (rot_min, rot_max)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data loaders
    print("Loading dataset...")
    train_loader, val_loader = get_data_loaders(
        args.data_dir, 
        batch_size=args.batch_size, 
        img_size=args.img_size,
        rotation_range=rotation_range
    )
    print(f"Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = get_model(pretrained=True, device=device)
    
    # Load pretrained model if provided
    if args.model_path:
        print(f"Loading pretrained model from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Train or evaluate
    if not args.eval_only:
        print(f"Starting training for {args.epochs} epochs...")
        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            save_dir=args.save_dir
        )
        
        # Plot training history
        plot_training_history(history, save_path=os.path.join(args.save_dir, 'training_history.png'))
    
    # Evaluate model
    print("Evaluating model...")
    test_mae, predictions = evaluate_model(model, val_loader, device)
    
    # Plot prediction results
    plot_prediction_results(
        predictions, 
        num_samples=10, 
        save_path=os.path.join(args.save_dir, 'prediction_results.png')
    )
    
    # Visualize predictions if requested
    if args.visualize:
        print("Visualizing predictions...")
        visualize_predictions_batch(model, val_loader, device)
        
        # Find an image to visualize rotation correction
        # Using the first image from validation set as an example
        try:
            sample_img_path = val_loader.dataset.dataset.image_paths[500]
            visualize_rotation_correction(model, sample_img_path, device, img_size=args.img_size)
        except Exception as e:
            print(f"Failed to visualize rotation correction: {e}")

if __name__ == "__main__":
    main()
