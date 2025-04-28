import torch
import torch.nn as nn
import torchvision.models as models

class RotationCorrectionModel(nn.Module):
    def __init__(self, pretrained=True):
        """
        CNN model to predict rotation correction angle
        
        Args:
            pretrained (bool): Whether to use pretrained ResNet weights
        """
        super(RotationCorrectionModel, self).__init__()
        
        # Use ResNet18 as the backbone
        resnet = models.resnet18(pretrained=pretrained)

        # Remove the classification layer and use the features
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Freeze all layers in the feature extractor
        for param in self.features.parameters():
            param.requires_grad = False

        # Unfreeze the last few layers
        for param in self.features[6:].parameters():  # Unfreeze the last few layers (adjust as needed)
            param.requires_grad = True

        # Regression head for angle prediction
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(resnet.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # Output is a single angle value
        )

        # self.regressor = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(resnet.fc.in_features, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 1)  # Output is a single angle value
        # )

    def forward(self, x):
        features = self.features(x)
        angle = self.regressor(features)
        return angle

def get_model(pretrained=True, device='cuda'):
    """
    Initialize the rotation correction model and move it to the specified device
    
    Args:
        pretrained (bool): Whether to use pretrained ResNet weights
        device (str): Device to move the model to ('cuda' or 'cpu')
        
    Returns:
        model: The initialized model
    """
    model = RotationCorrectionModel(pretrained=pretrained)
    model = model.to(device)
    return model
