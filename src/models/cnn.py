import torch
import torch.nn as nn

class FairCNN(nn.Module):
    """
    Fair CNN designed to match the parameter budget of the MLP (~112k parameters).
    Total parameters: 111,906.
    Includes feature extraction method for Q4 representations.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Stage 1: 32x32 -> 32x32
            nn.Conv2d(3, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            # Stage 2: 32x32 -> 16x16
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            # Stage 3: 16x16 -> 8x8
            nn.Conv2d(48, 72, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(72),
            nn.ReLU(inplace=True),
            # Stage 4: 8x8 -> 4x4
            nn.Conv2d(72, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            # Global Average Pooling: 4x4 -> 1x1
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten() # (B, 96)
        )
        
        # Penultimate representation layer (64-D)
        self.fc1 = nn.Linear(96, 64)
        self.relu = nn.ReLU(inplace=True)
        # Final classification layer
        self.fc2 = nn.Linear(64, num_classes)

    def extract_features(self, x):
        """
        Extracts the penultimate 64-dimensional representation.
        Used as the frozen representation for Q4 (SVM and Clustering).
        """
        feats = self.features(x)
        rep = self.relu(self.fc1(feats))
        return rep

    def forward(self, x):
        rep = self.extract_features(x)
        logits = self.fc2(rep)
        return logits

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
