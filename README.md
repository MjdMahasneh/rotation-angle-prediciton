# Image Rotation Correction Model

This project implements a deep learning model to predict the
rotation angle needed to correct an image to its upright position.
The model is trained on a dataset composed of 875 images of ranodom categorties with synthetic rotations applied.

The model uses a regression convolutional neural network (CNN) based
on ResNet18 architecture to learn the mapping between
rotated images and their corresponding rotation angles.



- **TODO:**

- [ ] expand the dataset.
- [ ] optimize the model.




## Project Structure

```
.
├── data_loader.py    # Data loading and preprocessing
├── model.py          # CNN model architecture
├── train.py          # Training and evaluation functions
├── visualization.py  # Functions for visualizing results
├── main.py           # Main script to train and test the model
├── README.md         # This file
└── checkpoints/      # Directory for saved models (created during training)
```

## Requirements

- Python
- PyTorch
- torchvision
- numpy
- matplotlib
- PIL
- tqdm








## How It Works

1. **Data Preparation**: Images are loaded from the dataset and randomly rotated by angles within a specified range (e.g., -45 to 45 degrees).
2. **Target Labels**: The target labels are the negative of the applied rotation angles, representing the angle needed to correct the rotation.
3. **Model Architecture**: A ResNet18-based CNN model is used to predict the rotation correction angle.
4. **Training**: The model is trained to minimize the mean squared error between predicted and actual correction angles.

## Usage

### Training

```bash
python main.py --data_dir ./path_to_images --batch_size 32 --epochs 20 --lr 0.001 --img_size 224 --rotation_range -45,45
```

### Evaluation

```bash
python main.py --data_dir ./path_to_images --eval_only --model_path ./checkpoints/best_model.pth --visualize
```

### Arguments

- `--data_dir`: Path to dataset directory
- `--batch_size`: Batch size for training
- `--epochs`: Number of epochs to train
- `--lr`: Learning rate
- `--img_size`: Image size to resize to
- `--rotation_range`: Range of random rotations (min,max)
- `--seed`: Random seed for reproducibility
- `--save_dir`: Directory to save models
- `--eval_only`: Only run evaluation (no training)
- `--model_path`: Path to a pretrained model
- `--visualize`: Visualize predictions

## Results

During training, the model's performance is tracked using:
- Mean Squared Error (MSE) loss
- Mean Absolute Error (MAE) in degrees

After training, visualizations are created to show:
- Training and validation loss curves
- True vs predicted rotation angles
- Error distribution
- Examples of rotation correction
